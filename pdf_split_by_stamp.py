#!/usr/bin/env python3
"""
Split a large PDF into smaller PDFs wherever a 'RECEIVED' stamp page appears.
Each stamped page becomes page 1 of its output file.

Only splits on stamps dated today or within the last 2 days.
Hardcode TESSERACT_CMD and POPPLER_PATH below if the tools are not on PATH.
Requires: pdf2image, pytesseract, pypdf, Pillow
"""

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    from pypdf import PdfReader, PdfWriter
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pdf2image pytesseract pypdf Pillow")
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
    # Manual formatting to avoid platform-specific %-d/%#d issues
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


def preprocess_for_ocr(image):
    """Preprocess the image to improve OCR accuracy on stamped pages."""
    # Crop to top 40% of page - stamps are always at the top
    w, h = image.size
    crop_top = image.crop((0, 0, w, int(h * 0.40)))

    # Convert to grayscale
    img = crop_top.convert("L")

    # Increase contrast aggressively
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)

    # Apply threshold to make text pure black/white
    img = img.point(lambda x: 0 if x < 128 else 255, mode="1")

    # Resize larger - Tesseract struggles with small text
    img = img.resize((img.width * 3, img.height * 3), Image.Resampling.LANCZOS)

    return img


def parse_date_from_text(text):
    """
    Try to find a date in OCR text and return a date object if valid.
    Handles formats like:
      SEP 02 2026
      SEP02 2026
      SEP 2 2026
    Returns None if no valid date found.
    """
    # Pattern: MONTH [optional space] DAY [space] YEAR
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
    """OCR a page image and return True if it has a recent RECEIVED stamp."""
    processed = preprocess_for_ocr(image)

    # Try multiple OCR configs
    configs = [
        "--psm 6",
        "--psm 11",
        "--psm 3",
    ]

    text = ""
    for cfg in configs:
        try:
            result = pytesseract.image_to_string(processed, config=cfg).upper()
            text += " " + result
        except Exception:
            pass

    if debug:
        print(f"  OCR text: {repr(text[:500])}")

    # Check if any extracted date falls within the 3-day window
    found_date = parse_date_from_text(text)
    has_by = "BY:" in text or "BY" in text or "SIGNATURE" in text
    has_received = "RECEIVED" in text
    if found_date and (has_by or has_received):
        month_str = found_date.strftime("%b").upper()
        day_str = str(found_date.day)
        year_str = str(found_date.year)
        date_str = f"{month_str} {day_str} {year_str}"
        date_str_nospace = f"{month_str}{day_str}{year_str}"
        if date_str in RECENT_DATES or date_str_nospace in RECENT_DATES_NOSPACE:
            if debug:
                print(f"  -> Recent date matched: {date_str}")
            return True

    # Fallback: look for RECEIVED + BY/SIGNATURE for non-dated stamps
    if has_received and has_by:
        return True

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
            print(f"\n  -> Stamp detected on page {i}")
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
        print("Usage: python pdf_split_by_stamp.py <input.pdf> [output_dir] [debug]")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "./split_output"
    debug = "--debug" in sys.argv or "debug" in sys.argv

    split_pdf(input_pdf, output_dir, debug=debug)
