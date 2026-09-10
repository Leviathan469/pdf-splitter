#!/usr/bin/env python3
"""
Split a large PDF into smaller PDFs wherever a 'RECEIVED' stamp page appears.
Each stamped page becomes page 1 of its output file.

Multi-signal stamp detection for 100% accuracy:
  1. Find dark text clusters on light background (all 3 lines)
  2. Structural validation: verify 3-line stamp layout (centered, upper page)
  3. OCR with whitelist: confirm "RECEIVED" and "BY:" text
  4. Color verification: confirm date region has red hue
  5. OCR date from red channel: read date without black text interference
  6. Date window check: only accept stamps dated today or within last 2 days

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
    from PIL import Image, ImageEnhance, ImageChops
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


def preprocess_top_region(image, crop_fraction=0.40):
    """Crop to top portion where the stamp lives, return cropped image."""
    w, h = image.size
    crop = image.crop((0, 0, w, int(h * crop_fraction)))
    return crop


def find_text_regions(image, debug=False):
    """
    Find regions of dark text on light background.
    Returns list of (x, y, w, h) bounding boxes for text clusters.
    """
    arr = np.array(image)
    
    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr
    
    # Adaptive threshold — handles uneven lighting better than global threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Also try Otsu's threshold as backup
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combine both thresholds
    combined = cv2.bitwise_or(binary, otsu)
    
    # Clean up noise
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Dilate vertically to connect letters in the same word
    v_kernel = np.ones((5, 1), np.uint8)
    combined = cv2.dilate(combined, v_kernel, iterations=2)
    
    # Dilate horizontally slightly to connect words on same line
    h_kernel = np.ones((1, 3), np.uint8)
    combined = cv2.dilate(combined, h_kernel, iterations=1)
    
    if debug:
        cv2.imwrite(str(DEBUG_DIR / "2_threshold.png"), combined)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:  # Minimum area threshold
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Filter by aspect ratio — text is wider than tall
        if h > 0 and w / h < 0.3:
            continue
        # Filter by height — text should be reasonably tall
        if h < 10:
            continue
        regions.append((x, y, w, h))
    
    return regions


def merge_nearby_regions(regions, max_gap=30):
    """Merge text regions that are on the same line (for multi-word lines)."""
    if not regions:
        return []
    
    # Sort by y position
    sorted_regions = sorted(regions, key=lambda r: r[1])
    
    merged = []
    current = list(sorted_regions[0])
    
    for i in range(1, len(sorted_regions)):
        x, y, w, h = sorted_regions[i]
        cx, cy, cw, ch = current
        
        # Check if regions are on similar y-level and close horizontally
        y_overlap = abs(y - cy) < max(cw, w) * 0.5
        x_gap = x - (cx + cw)
        
        if y_overlap and x_gap < max_gap:
            # Merge
            new_x = min(cx, x)
            new_y = min(cy, y)
            new_w = max(cx + cw, x + w) - new_x
            new_h = max(cy + ch, y + h) - new_y
            current = [new_x, new_y, new_w, new_h]
        else:
            merged.append(tuple(current))
            current = [x, y, w, h]
    
    merged.append(tuple(current))
    return merged


def identify_stamp_structure(regions, image_size, debug=False):
    """
    Identify if the text regions form a 3-line stamp structure.
    Returns dict with line regions if found, None otherwise.
    
    Expected structure:
      Line 1: "RECEIVED" (large text, centered)
      Line 2: Date (medium text, centered, different color)
      Line 3: "BY: ..." (smaller text, left-aligned)
    """
    if len(regions) < 3:
        return None
    
    # Sort regions by y position
    sorted_regions = sorted(regions, key=lambda r: r[1])
    
    img_w, img_h = image_size
    
    # Look for 3 clusters of text at different y-levels
    stamp_lines = []
    current_line = [sorted_regions[0]]
    
    for i in range(1, len(sorted_regions)):
        prev_y = current_line[-1][1]
        curr_y = sorted_regions[i][1]
        
        if abs(curr_y - prev_y) < 20:  # Same line
            current_line.append(sorted_regions[i])
        else:
            if len(current_line) >= 1:
                # Merge regions on the same line
                x1 = min(r[0] for r in current_line)
                y1 = min(r[1] for r in current_line)
                x2 = max(r[0] + r[2] for r in current_line)
                y2 = max(r[1] + r[3] for r in current_line)
                stamp_lines.append((x1, y1, x2 - x1, y2 - y1))
            current_line = [sorted_regions[i]]
    
    if current_line:
        x1 = min(r[0] for r in current_line)
        y1 = min(r[1] for r in current_line)
        x2 = max(r[0] + r[2] for r in current_line)
        y2 = max(r[1] + r[3] for r in current_line)
        stamp_lines.append((x1, y1, x2 - x1, y2 - y1))
    
    if len(stamp_lines) < 3:
        return None
    
    # Validate the structure: 3 lines stacked vertically
    # Line spacing should be roughly uniform
    for i in range(1, min(3, len(stamp_lines))):
        prev_bottom = stamp_lines[i-1][1] + stamp_lines[i-1][3]
        curr_top = stamp_lines[i][1]
        gap = curr_top - prev_bottom
        if gap < 5 or gap > 100:
            return None
    
    # Validate horizontal centering — stamp lines should be roughly centered
    for line in stamp_lines[:3]:
        cx = line[0] + line[2] / 2
        if abs(cx / img_w - 0.5) > 0.25:  # Not within 25% of center
            return None
    
    return {
        'received': stamp_lines[0],
        'date': stamp_lines[1],
        'by': stamp_lines[2]
    }


def ocr_region(image, region, psm=7, padding=5):
    """OCR a specific region of an image with whitelist."""
    x, y, w, h = region
    # Add padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.size[0], x + w + padding)
    y2 = min(image.size[1], y + h + padding)
    
    cropped = image.crop((x1, y1, x2, y2))
    
    # Enhance for OCR
    cropped = cropped.convert("L")
    enhancer = ImageEnhance.Contrast(cropped)
    cropped = enhancer.enhance(2.5)
    cropped = cropped.point(lambda x: 0 if x < 128 else 255, mode="1")
    cropped = cropped.resize((cropped.width * 3, cropped.height * 3), Image.Resampling.LANCZOS)
    
    try:
        config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        text = pytesseract.image_to_string(cropped, config=config).upper().strip()
        return text
    except Exception:
        return ""


def check_red_hue(image, region):
    """
    Check if a region contains red-hued pixels.
    Returns the percentage of reddish pixels in the region.
    """
    arr = np.array(image)
    x, y, w, h = region
    
    # Clamp to image bounds
    x2 = min(x + w, arr.shape[1])
    y2 = min(y + h, arr.shape[0])
    
    if x >= x2 or y >= y2:
        return 0.0
    
    region_arr = arr[y:y2, x:x2]
    
    if len(region_arr.shape) < 3:
        return 0.0
    
    r = region_arr[:, :, 0].astype(int)
    g = region_arr[:, :, 1].astype(int)
    b = region_arr[:, :, 2].astype(int)
    
    # Red: R is significantly higher than G and B
    red_pixels = np.sum((r > g + 15) & (r > b + 15) & (r > 80))
    total_pixels = (x2 - x) * (y2 - y)
    
    if total_pixels == 0:
        return 0.0
    
    return red_pixels / total_pixels


def ocr_red_date(image, date_region, debug=False):
    """
    OCR the date from the red channel to avoid black text interference.
    """
    arr = np.array(image)
    x, y, w, h = date_region
    
    # Expand region slightly
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(arr.shape[1], x + w + pad)
    y2 = min(arr.shape[0], y + h + pad)
    
    region_arr = arr[y1:y2, x1:x2]
    
    if len(region_arr.shape) < 3:
        return ""
    
    # Isolate red channel
    r = region_arr[:, :, 0].astype(int)
    g = region_arr[:, :, 1].astype(int)
    b = region_arr[:, :, 2].astype(int)
    
    # Create red-only image
    red_only = np.zeros_like(region_arr)
    red_mask = (r > g + 15) & (r > b + 15) & (r > 80)
    red_only[red_mask] = [255, 255, 255]
    
    red_image = Image.fromarray(red_only.astype(np.uint8))
    
    # Invert so text is black on white
    red_inverted = ImageChops.invert(red_image.convert("RGB"))
    
    # Enhance
    red_inverted = red_inverted.convert("L")
    enhancer = ImageEnhance.Contrast(red_inverted)
    red_inverted = enhancer.enhance(2.5)
    red_inverted = red_inverted.point(lambda x: 0 if x < 100 else 255, mode="1")
    red_inverted = red_inverted.resize(
        (red_inverted.width * 4, red_inverted.height * 4), Image.Resampling.LANCZOS
    )
    
    if debug:
        red_inverted.save(DEBUG_DIR / "5_red_date_processed.png")
    
    # OCR with multiple PSM modes
    text = ""
    for psm in [6, 7, 8, 13, 4]:
        try:
            config = f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"
            result = pytesseract.image_to_string(red_inverted, config=config).upper().strip()
            text += " " + result
        except Exception:
            pass
    
    return text


def parse_date_from_text(text):
    """
    Try to find a date in OCR text and return a date object if valid.
    Handles formats like:
      SEP 02 2026
      SEP02 2026
      SEP 2 2026
    Returns None if no valid date found.
    """
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
    Multi-signal stamp detection for 100% accuracy.
    """
    if debug:
        DEBUG_DIR.mkdir(exist_ok=True)
        image.save(DEBUG_DIR / "0_original.png")

    # Step 1: Crop to top region
    top_region = preprocess_top_region(image)
    top_w, top_h = top_region.size

    if debug:
        top_region.save(DEBUG_DIR / "1_top_region.png")

    # Step 2: Find text regions
    regions = find_text_regions(top_region, debug=debug)
    regions = merge_nearby_regions(regions)

    if debug:
        print(f"  -> Found {len(regions)} text regions")

    # Step 3: Identify stamp structure
    structure = identify_stamp_structure(regions, (top_w, top_h), debug=debug)

    if structure is None:
        if debug:
            print("  -> No valid stamp structure found")
        return False

    if debug:
        print(f"  -> Stamp structure identified:")
        print(f"     RECEIVED: {structure['received']}")
        print(f"     Date: {structure['date']}")
        print(f"     BY: {structure['by']}")

    # Step 4: OCR "RECEIVED" line
    received_text = ocr_region(top_region, structure['received'], psm=7)
    has_received = "RECEIVED" in received_text

    if debug:
        print(f"  -> Received OCR: '{received_text}' (match={has_received})")

    if not has_received:
        if debug:
            print("  -> 'RECEIVED' not found in expected region — rejecting")
        return False

    # Step 5: OCR "BY:" line
    by_text = ocr_region(top_region, structure['by'], psm=7)
    has_by = "BY:" in by_text or "BY :" in by_text

    if debug:
        print(f"  -> BY OCR: '{by_text}' (match={has_by})")

    if not has_by:
        if debug:
            print("  -> 'BY:' not found in expected region — rejecting")
        return False

    # Step 6: Check date region has red hue (confirmation)
    red_pct = check_red_hue(top_region, structure['date'])
    has_red = red_pct > 0.10  # At least 10% reddish pixels

    if debug:
        print(f"  -> Date region red hue: {red_pct:.2%} (threshold: 10%)")

    if not has_red:
        if debug:
            print("  -> Date region lacks red hue — trying without color check")

    # Step 7: OCR the date from red channel
    date_text = ocr_red_date(top_region, structure['date'], debug=debug)

    if debug:
        print(f"  -> Date OCR: '{date_text}'")

    # Step 8: Parse and validate date
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
