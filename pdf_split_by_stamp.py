#!/usr/bin/env python3
"""
Split a large PDF into smaller PDFs wherever a 'RECEIVED' stamp page appears.
Each stamped page becomes page 1 of its output file.

Stamp detection strategy (for 100% accuracy):
  1. Find the red date using color isolation (most distinctive feature)
  2. Calculate exact positions of stamp elements relative to the date
  3. Extract and OCR "RECEIVED" line (above date, outlined text)
  4. Extract and OCR "BY:" line (below date)
  5. OCR the red-isolated date channel to read the date
  6. Verify all signals agree and date is within window

Requires: pdf2image, pytesseract, pypdf, Pillow, opencv-python, numpy
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

# ---------------------------------------------------------------------------
# CONFIGURE THESE if IT installs to non-standard locations:
# ---------------------------------------------------------------------------
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")
POPPLER_PATH = os.environ.get("POPPLER_PATH", "")
# Examples:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# POPPLER_PATH = r"C:\poppler\Library\bin"
# ---------------------------------------------------------------------------

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Build the 3-day window: today + last 2 days
TODAY = date.today()
RECENT_DATES = set()
RECENT_DATES_NOSPACE = set()
for i in range(3):
    d = TODAY - timedelta(days=i)
    month_str = d.strftime("%b").upper()
    day_str = str(d.day)
    year_str = str(d.year)
    date_str = f"{month_str} {day_str} {year_str}"
    RECENT_DATES.add(date_str)
    RECENT_DATES_NOSPACE.add(f"{month_str}{day_str}{year_str}")
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Tesseract whitelist — only characters found in a stamp
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:./- "

# Debug output directory
DEBUG_DIR = Path("./stamp_debug")


def find_red_date_region(image, debug=False):
    """
    Find the red date region using color isolation.
    Returns (x, y, w, h) of the combined red region, or None.
    """
    arr = np.array(image)
    if len(arr.shape) < 3:
        return None
    
    r = arr[:, :, 0].astype(int)
    g = arr[:, :, 1].astype(int)
    b = arr[:, :, 2].astype(int)
    
    red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    if debug:
        cv2.imwrite(str(DEBUG_DIR / "1_red_mask.png"), red_mask)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    h_img, w_img = red_mask.shape
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if y > h_img * 0.8:
            continue
        valid.append((x, y, w, h))
    
    if not valid:
        return None
    
    min_x = min(v[0] for v in valid)
    min_y = min(v[1] for v in valid)
    max_x = max(v[0]+v[2] for v in valid)
    max_y = max(v[1]+v[3] for v in valid)
    
    pad = 10
    return (max(0, min_x-pad), max(0, min_y-pad), 
            min(w_img, max_x+pad)-max(0, min_x-pad), 
            min(h_img, max_y+pad)-max(0, min_y-pad))


def get_stamp_element_regions(date_region, image_size):
    """
    Given the date region, calculate the expected positions of other elements.
    
    Returns dict with regions for 'received', 'date', 'by' in page coordinates.
    """
    dx, dy, dw, dh = date_region
    img_w, img_h = image_size
    
    # RECEIVED is above the date, roughly same width, 1.5x date height
    recv_h = int(dh * 1.5)
    recv_w = int(dw * 2.5)
    recv_x = dx + dw//2 - recv_w//2
    recv_y = max(0, dy - int(dh * 1.8))
    
    # BY is below the date, roughly same width, 0.8x date height
    by_h = int(dh * 0.8)
    by_w = int(dw * 2.0)
    by_x = dx - int(dw * 0.3)  # BY is left-aligned relative to date
    by_y = min(img_h - by_h, dy + dh + int(dh * 0.3))
    
    return {
        'received': (recv_x, recv_y, recv_w, recv_h),
        'date': date_region,
        'by': (by_x, by_y, by_w, by_h)
    }


def ocr_received(image, region, debug=False):
    """
    OCR the RECEIVED line. This text is often outlined/hollow, so we use
    fuzzy matching since Tesseract may misread individual characters.
    """
    x, y, w, h = region
    img_w, img_h = image.size
    
    # Add generous padding
    pad = 15
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    
    cropped = image.crop((x1, y1, x2, y2))
    
    if debug:
        cropped.save(DEBUG_DIR / "2_received_crop.png")
    
    # Strategy: use edge detection + fill to solidify outlined text
    gray = np.array(cropped.convert("L"))
    
    best_text = ""
    
    # Method 1: Aggressive morphological close to fill outlined text
    for thresh_val in [180, 190, 200]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Close gaps in outlined text
        for kernel_size in [3, 5, 7]:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
            filled = cv2.dilate(filled, np.ones((2,2), np.uint8), iterations=1)
            
            # Invert for Tesseract (black text on white)
            result = 255 - filled
            result_img = Image.fromarray(result)
            result_img = result_img.resize(
                (result_img.width * 4, result_img.height * 4),
                Image.Resampling.LANCZOS
            )
            
            for psm in [6, 7, 8]:
                try:
                    config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
                    text = pytesseract.image_to_string(result_img, config=config).upper().strip()
                    if text:
                        best_text += " " + text
                except Exception:
                    pass
    
    # Method 2: Direct OCR on grayscale with high contrast
    cropped_gray = cropped.convert("L")
    enhancer = ImageEnhance.Contrast(cropped_gray)
    cropped_gray = enhancer.enhance(3.0)
    cropped_gray = cropped_gray.resize(
        (cropped_gray.width * 4, cropped_gray.height * 4),
        Image.Resampling.LANCZOS
    )
    
    for psm in [6, 7, 8, 13]:
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            text = pytesseract.image_to_string(cropped_gray, config=config).upper().strip()
            if text:
                best_text += " " + text
        except Exception:
            pass
    
    # Method 3: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    adaptive_inv = 255 - adaptive
    adaptive_img = Image.fromarray(adaptive_inv)
    adaptive_img = adaptive_img.resize(
        (adaptive_img.width * 4, adaptive_img.height * 4),
        Image.Resampling.LANCZOS
    )
    
    for psm in [6, 7]:
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            text = pytesseract.image_to_string(adaptive_img, config=config).upper().strip()
            if text:
                best_text += " " + text
        except Exception:
            pass
    
    return best_text


def is_close_to_received(text):
    """
    Check if text contains a word that's close to "RECEIVED".
    Uses edit distance to handle OCR errors from outlined fonts.
    """
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
    
    words = text.split()
    for word in words:
        # Only check words of similar length (7-9 chars for "RECEIVED")
        if 7 <= len(word) <= 9:
            dist = edit_distance(word, "RECEIVED")
            if dist <= 2:  # Allow up to 2 character errors
                return True
    return False


def ocr_by(image, region, debug=False):
    """OCR the BY: line (solid text, should be easier)."""
    x, y, w, h = region
    img_w, img_h = image.size
    
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)
    
    cropped = image.crop((x1, y1, x2, y2))
    
    if debug:
        cropped.save(DEBUG_DIR / "3_by_crop.png")
    
    best_text = ""
    
    # Multiple preprocessing approaches
    gray = np.array(cropped.convert("L"))
    
    for thresh in [128, 150, 180, 200]:
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
        binary_pil = Image.fromarray(255 - binary)
        binary_pil = binary_pil.resize(
            (binary_pil.width * 4, binary_pil.height * 4),
            Image.Resampling.LANCZOS
        )
        
        for psm in [6, 7, 8, 13]:
            try:
                config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
                text = pytesseract.image_to_string(binary_pil, config=config).upper().strip()
                if text:
                    best_text += " " + text
            except Exception:
                pass
    
    return best_text


def ocr_red_date(image, date_region, debug=False):
    """OCR the date from red channel isolation."""
    arr = np.array(image)
    if len(arr.shape) < 3:
        return ""
    
    x, y, w, h = date_region
    pad = 10
    x1, y1 = max(0, x-pad), max(0, y-pad)
    x2, y2 = min(arr.shape[1], x+w+pad), min(arr.shape[0], y+h+pad)
    
    region = arr[y1:y2, x1:x2]
    
    r = region[:,:,0].astype(int)
    g = region[:,:,1].astype(int)
    b = region[:,:,2].astype(int)
    
    red_only = np.zeros_like(region)
    red_mask = (r > g+15) & (r > b+15) & (r > 80)
    red_only[red_mask] = [255, 255, 255]
    
    red_img = Image.fromarray(red_only.astype(np.uint8))
    red_inv = ImageChops.invert(red_img.convert("RGB"))
    
    red_inv = red_inv.convert("L")
    enhancer = ImageEnhance.Contrast(red_inv)
    red_inv = enhancer.enhance(2.5)
    red_inv = red_inv.point(lambda x: 0 if x < 100 else 255, mode="1")
    red_inv = red_inv.resize((red_inv.width*4, red_inv.height*4), Image.Resampling.LANCZOS)
    
    if debug:
        red_inv.save(DEBUG_DIR / "4_red_date.png")
    
    text = ""
    for psm in [6, 7, 8, 13]:
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            result = pytesseract.image_to_string(red_inv, config=config).upper().strip()
            text += " " + result
        except Exception:
            pass
    
    return text


def parse_date_from_text(text):
    """Parse date from text. Returns date object or None."""
    matches = re.findall(
        r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{1,2})\s*(\d{4})\b",
        text
    )
    for month_str, day_str, year_str in matches:
        try:
            month = MONTH_MAP.get(month_str.upper())
            day = int(day_str)
            year = int(year_str)
            if month and 1 <= day <= 31 and 2000 <= year <= 2100:
                return date(year, month, day)
        except ValueError:
            continue
    return None


def page_has_stamp(image, debug=False) -> bool:
    """
    Multi-signal stamp detection.
    All of the following must be true:
      1. Red date region found
      2. "RECEIVED" OCR'd from expected region
      3. "BY:" OCR'd from expected region  
      4. Date parsed from red channel
      5. Date within window
    """
    if debug:
        DEBUG_DIR.mkdir(exist_ok=True)
        image.save(DEBUG_DIR / "0_original.png")
    
    # 1. Find red date
    date_region = find_red_date_region(image, debug=debug)
    if date_region is None:
        if debug:
            print("  -> No red date region found")
        return False
    
    if debug:
        print(f"  -> Red date region: x={date_region[0]}, y={date_region[1]}, w={date_region[2]}, h={date_region[3]}")
    
    # 2. Get all element regions
    regions = get_stamp_element_regions(date_region, image.size)
    
    # 3. OCR RECEIVED (with fuzzy matching for outlined text)
    received_text = ocr_received(image, regions['received'], debug=debug)
    has_received = "RECEIVED" in received_text or is_close_to_received(received_text)
    
    if debug:
        print(f"  -> Received OCR: {repr(received_text[:300])}")
        print(f"  -> has_received={has_received}")
    
    if not has_received:
        if debug:
            print("  -> 'RECEIVED' not found — rejecting")
        return False
    
    # 4. OCR BY
    by_text = ocr_by(image, regions['by'], debug=debug)
    has_by = "BY:" in by_text or "BY :" in by_text or "BY" in by_text
    
    if debug:
        print(f"  -> BY OCR: {repr(by_text[:200])}")
        print(f"  -> has_by={has_by}")
    
    if not has_by:
        if debug:
            print("  -> 'BY:' not found — rejecting")
        return False
    
    # 5. OCR date from red channel
    date_text = ocr_red_date(image, regions['date'], debug=debug)
    
    if debug:
        print(f"  -> Date OCR: {repr(date_text)}")
    
    # 6. Parse and validate date
    found_date = parse_date_from_text(date_text)
    if found_date is None:
        if debug:
            print("  -> Could not parse date — rejecting")
        return False
    
    month_str = found_date.strftime("%b").upper()
    day_str = str(found_date.day)
    year_str = str(found_date.year)
    date_str = f"{month_str} {day_str} {year_str}"
    date_str_nospace = f"{month_str}{day_str}{year_str}"
    
    if date_str in RECENT_DATES or date_str_nospace in RECENT_DATES_NOSPACE:
        if debug:
            print(f"  ✅ STAMP CONFIRMED: date={date_str}")
        return True
    
    if debug:
        print(f"  -> Date {date_str} not in window — rejecting")
    return False


def split_pdf(input_pdf: str, output_dir: str, dpi: int = 300, debug: bool = False):
    """Split PDF at stamped pages."""
    input_path = Path(input_pdf)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading PDF: {input_path}")
    reader = PdfReader(str(input_path))
    total = len(reader.pages)
    print(f"Total pages: {total}")
    print(f"Looking for stamps dated: {sorted(RECENT_DATES)}")

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
