#!/usr/bin/env python3
"""
pdf_split_by_stamp.py

Split PDFs at RECEIVED stamps using red channel detection.

Strategy:
  1. Find stamp by detecting red ink (the date)
  2. Validate by structural properties (size, position, aspect ratio)
  3. Confirm stamp exists - no OCR needed for detection

Red ink is rare on insurance documents. The probability of random red ink
forming the exact shape of a date (wide aspect ratio, specific size range,
upper portion of page) is essentially zero.

Usage:
  python pdf_split_by_stamp.py <input.pdf> [output_dir] [--debug]
"""

import os
import re
import sys
import cv2
import numpy as np
from datetime import date, timedelta
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image, ImageEnhance, ImageChops, ImageFilter
    from pypdf import PdfReader, PdfWriter
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pdf2image pytesseract pypdf Pillow opencv-python numpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")
POPPLER_PATH = os.environ.get("POPPLER_PATH", "")

TODAY = date.today()
RECENT_DATES = set()
for i in range(3):
    d = TODAY - timedelta(days=i)
    RECENT_DATES.add(f"{d.strftime('%b').upper()} {d.day} {d.year}")

MONTH_MAP = {name: i+1 for i, name in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

DEBUG_DIR = Path("./stamp_debug")

# ============================================================================

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def find_stamp_area(image, debug=False):
    """
    Find the stamp area by detecting red ink (the date).
    Validates the stamp by structural properties.
    
    Returns (x, y, w, h) of stamp area, or None if no valid stamp found.
    """
    arr = np.array(image)
    if len(arr.shape) < 3:
        return None

    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    
    # Detect red pixels (R significantly higher than G and B)
    red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    if debug:
        DEBUG_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(DEBUG_DIR / "1_red_mask.png"), red_mask)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    h_img, w_img = red_mask.shape
    min_area = h_img * w_img * 0.0001
    
    valid = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > h_img * 0.8:
            continue
        if h > 0 and w / h < 0.8:
            continue
        valid.append((x, y, w, h))
    
    if not valid:
        return None
    
    # Group nearby contours
    groups = []
    used = set()
    for i, (x1, y1, w1, h1) in enumerate(valid):
        if i in used:
            continue
        group = [(x1, y1, w1, h1)]
        used.add(i)
        for j, (x2, y2, w2, h2) in enumerate(valid):
            if j in used or i == j:
                continue
            if abs(y1 - y2) < h1 * 2 and abs(x1 - x2) < h1 * 3:
                group.append((x2, y2, w2, h2))
                used.add(j)
        groups.append(group)
    
    # Validate each group
    for group in groups:
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[2] for b in group]
        hs = [b[3] for b in group]
        
        date_h = max(y + h for y, h in zip(ys, hs)) - min(ys)
        date_w = max(x + w for x, w in zip(xs, ws)) - min(xs)
        
        # Structural validation
        if date_h < 15 or date_h > 150:
            continue
        if date_w / date_h < 1.5:
            continue
        
        # Expand to full stamp area
        min_x = max(0, min(xs) - int(date_h * 2))
        min_y = max(0, min(ys) - int(date_h * 3))
        max_x = min(w_img, max(x + w for x, w in zip(xs, ws)) + int(date_h * 2))
        max_y = min(h_img, max(y + h for y, h in zip(ys, hs)) + int(date_h * 2))
        
        stamp_w = max_x - min_x
        stamp_h = max_y - min_y
        
        if stamp_h > 0:
            aspect = stamp_w / stamp_h
            if aspect < 0.5 or aspect > 3.0:
                continue
        
        return (min_x, min_y, stamp_w, stamp_h)
    
    return None


def page_has_stamp(image, debug=False) -> bool:
    """
    Detect stamp by finding red ink with stamp-like structure.
    No OCR needed - red ink + structure is sufficient.
    """
    if debug:
        DEBUG_DIR.mkdir(exist_ok=True)
        image.save(DEBUG_DIR / "0_original.png")
    
    stamp_area = find_stamp_area(image, debug=debug)
    if stamp_area is None:
        if debug:
            print("  -> No valid stamp area found")
        return False
    
    if debug:
        print(f"  -> Stamp area: x={stamp_area[0]}, y={stamp_area[1]}, w={stamp_area[2]}, h={stamp_area[3]}")
        print(f"  ✅ STAMP CONFIRMED (red ink + structure)")
    
    return True


def split_pdf(input_pdf: str, output_dir: str, dpi: int = 300, debug: bool = False):
    """Split PDF at stamped pages."""
    input_path = Path(input_pdf)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading PDF: {input_path}")
    reader = PdfReader(str(input_path))
    total = len(reader.pages)
    print(f"Total pages: {total}")

    convert_kwargs = {"dpi": dpi, "fmt": "png", "thread_count": 4}
    if POPPLER_PATH:
        convert_kwargs["poppler_path"] = POPPLER_PATH

    print(f"Converting pages to images (dpi={dpi})...")
    images = convert_from_path(str(input_path), **convert_kwargs)
    assert len(images) == total, f"Image count {len(images)} != page count {total}"

    stamp_pages = []
    for i, img in enumerate(images, start=1):
        print(f"Scanning page {i}/{total}...", end="\r")
        if page_has_stamp(img, debug=debug):
            stamp_pages.append(i)
            print(f"\n  -> Stamp DETECTED on page {i}")
        elif debug:
            print(f"\n  -> No stamp on page {i}")

    print(f"\nFound {len(stamp_pages)} stamp pages: {stamp_pages}")

    if not stamp_pages:
        print("No recent stamp pages found. Exiting.")
        return

    split_groups = []
    for idx, stamp_page in enumerate(stamp_pages):
        start = stamp_page
        end = stamp_pages[idx + 1] - 1 if idx + 1 < len(stamp_pages) else total
        split_groups.append((start, end))

    print(f"Split into {len(split_groups)} PDF(s):")
    for start, end in split_groups:
        print(f"  Pages {start}-{end}")

    stem = input_path.stem
    for i, (start, end) in enumerate(split_groups, start=1):
        writer = PdfWriter()
        for page_num in range(start, end + 1):
            writer.add_page(reader.pages[page_num - 1])

        out_name = out_dir / f"{stem}_part{i:03d}_p{start}-{end}.pdf"
        with open(out_name, "wb") as f:
            writer.write(f)
        print(f"  Wrote: {out_name}  ({end - start + 1} pages)")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_split_by_stamp.py <input.pdf> [output_dir] [--debug]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "./split_output"
    debug = "--debug" in sys.argv

    split_pdf(input_pdf, output_dir, debug=debug)
