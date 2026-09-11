#!/usr/bin/env python3
"""
Optimized PDF Splitter - faster without losing accuracy.

Key optimizations:
1. Pre-filter: Quick red ink check rejects non-stamp pages early
2. Reduced scale range: Only search near 1.0 (where real stamps match)
3. Grayscale matching: Already using, but ensured
4. Skip obvious non-stamps: If no red ink in upper-left, skip template matching
"""
import argparse
import cv2
import numpy as np
import subprocess
import sys
from pathlib import Path

try:
    from pypdf import PdfReader, PdfWriter
except ImportError:
    print("ERROR: pypdf not installed. Run: pip install pypdf")
    sys.exit(1)


def extract_pages(pdf_path, output_dir, dpi=300):
    """Convert PDF pages to PNG images using pdftoppm."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdftoppm = find_pdftoppm()
    if pdftoppm is None:
        print("ERROR: pdftoppm not found. Install Poppler or add to PATH.")
        sys.exit(1)
    
    print(f"Extracting pages from {pdf_path}...")
    result = subprocess.run([
        str(pdftoppm), "-png", "-r", str(dpi),
        str(pdf_path),
        str(output_dir / "page")
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: pdftoppm failed: {result.stderr}")
        sys.exit(1)
    
    pages = sorted(output_dir.glob("page-*.png"))
    print(f"  Extracted {len(pages)} pages")
    return pages


def find_pdftoppm():
    """Find pdftoppm executable."""
    paths = [
        "pdftoppm",
        r"C:\Users\acollazo\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin\pdftoppm.exe"
    ]
    
    for p in paths:
        try:
            result = subprocess.run([p, "-v"], capture_output=True, text=True)
            if result.returncode == 0 or "pdftoppm" in result.stderr:
                return p
        except FileNotFoundError:
            continue
    
    return None


def quick_precheck(page_img):
    """
    Fast pre-check: does this page have red ink in the upper-left?
    Returns True if page might have a stamp, False if definitely not.
    """
    h, w = page_img.shape[:2]
    
    # Only check upper-left quadrant (where stamps are)
    upper_left = page_img[0:int(h*0.4), 0:int(w*0.6)]
    
    # Convert to HSV for color detection
    hsv = cv2.cvtColor(upper_left, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 | mask2
    
    # Count red pixels
    red_pixels = np.sum(red_mask > 0)
    
    # Need at least some red pixels to be a stamp page
    return red_pixels > 50  # Very low threshold, just to skip blank pages


def detect_stamp(page_img, template, threshold=0.35):
    """
    Detect if stamp exists in page image using template matching.
    
    Key insight: real stamps match at scale ~1.0, false positives at wrong scales.
    We boost confidence for matches near scale=1.0 and penalize others.
    
    Returns: (found, confidence, scale)
    """
    page_gray = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Quick pre-check first
    if not quick_precheck(page_img):
        return False, 0.0, 0.0
    
    # Search scales from 0.7 to 1.3 (narrower range, real stamps are ~1.0)
    scales = np.arange(0.7, 1.35, 0.05)
    
    best_confidence = 0
    best_scale = 0
    
    for scale in scales:
        h, w = template_gray.shape
        resized = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
        
        if resized.shape[0] > page_gray.shape[0] or resized.shape[1] > page_gray.shape[1]:
            continue
        
        result = cv2.matchTemplate(page_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > best_confidence:
            best_confidence = max_val
            best_scale = scale
    
    # Apply scale-based confidence adjustment
    scale_penalty = abs(best_scale - 1.0)
    
    if scale_penalty < 0.15:
        adjusted_confidence = best_confidence * 1.2
    elif scale_penalty < 0.3:
        adjusted_confidence = best_confidence
    else:
        adjusted_confidence = best_confidence * 0.7
    
    found = adjusted_confidence >= threshold
    return found, adjusted_confidence, best_scale


def split_pdf_by_stamps(pdf_path, template, threshold=0.35, dpi=300, work_dir=None):
    """Split PDF into multiple PDFs based on stamp detection."""
    pdf_path = Path(pdf_path)
    
    if work_dir is None:
        work_dir = pdf_path.parent / "split_work"
    else:
        work_dir = Path(work_dir)
    
    pages = extract_pages(pdf_path, work_dir, dpi)
    
    if not pages:
        print("ERROR: No pages extracted from PDF")
        return []
    
    print(f"\nDetecting stamps on {len(pages)} pages...")
    stamp_pages = []
    precheck_passed = 0
    
    for i, page_path in enumerate(pages):
        page_img = cv2.imread(str(page_path))
        found, confidence, scale = detect_stamp(page_img, template, threshold)
        
        status = "✓ STAMP" if found else "✗ None"
        print(f"  Page {i+1:3d} ({page_path.name:30s}): {status:8s} (conf={confidence:.2f}, scale={scale:.2f})")
        
        if found:
            stamp_pages.append(i)
    
    if not stamp_pages:
        print("\nWARNING: No stamps detected. PDF will not be split.")
        return [list(range(len(pages)))]
    
    ranges = []
    for i, stamp_idx in enumerate(stamp_pages):
        start = stamp_idx
        end = stamp_pages[i + 1] if i + 1 < len(stamp_pages) else len(pages)
        ranges.append(list(range(start, end)))
    
    print(f"\n{'='*60}")
    print(f"SPLIT RESULTS")
    print(f"{'='*60}")
    print(f"Total pages: {len(pages)}")
    print(f"Stamps found on pages: {[p+1 for p in stamp_pages]}")
    print(f"Number of output PDFs: {len(ranges)}")
    for i, r in enumerate(ranges):
        print(f"  PDF {i+1}: Pages {r[0]+1}-{r[-1]+1} ({len(r)} pages)")
    
    return ranges


def create_split_pdfs(pdf_path, ranges, output_dir):
    """Create split PDF files from page ranges."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating split PDFs...")
    
    reader = PdfReader(str(pdf_path))
    
    output_files = []
    for i, page_range in enumerate(ranges):
        writer = PdfWriter()
        
        for page_idx in page_range:
            writer.add_page(reader.pages[page_idx])
        
        stem = pdf_path.stem
        output_path = output_dir / f"{stem}_part{i+1:03d}.pdf"
        
        with open(output_path, 'wb') as f:
            writer.write(f)
        
        output_files.append(output_path)
        print(f"  Created: {output_path.name} ({len(page_range)} pages)")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(description="Split PDF by RECEIVED stamp detection")
    parser.add_argument("pdf", help="Input PDF file path")
    parser.add_argument("--template", required=True, help="Stamp template image path")
    parser.add_argument("--threshold", type=float, default=0.35, help="Detection threshold (0-1)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering")
    parser.add_argument("--output-dir", required=True, help="Output directory for split PDFs")
    
    args = parser.parse_args()
    
    template = cv2.imread(str(args.template))
    if template is None:
        print(f"ERROR: Could not load template from {args.template}")
        sys.exit(1)
    
    print(f"Template loaded: {template.shape}")
    
    work_dir = Path(args.output_dir) / "temp_pages"
    
    ranges = split_pdf_by_stamps(
        args.pdf,
        template,
        threshold=args.threshold,
        dpi=args.dpi,
        work_dir=work_dir
    )
    
    if not ranges:
        print("ERROR: Could not determine page ranges")
        sys.exit(1)
    
    output_files = create_split_pdfs(args.pdf, ranges, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Created {len(output_files)} PDF files in {args.output_dir}")
    
    import shutil
    if work_dir.exists():
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
