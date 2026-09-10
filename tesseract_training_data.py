#!/usr/bin/env python3
"""
tesseract_training_data.py

Auto-generate Tesseract training data from a multi-stamp scan.

Workflow:
  1. Scan a page with as many stamps as you can fit (vary the dates!)
  2. Run this script on the scan
  3. It finds each stamp's red date region
  4. Extracts stamp images and generates ground truth files
  5. Produces a training-ready dataset

Usage:
  python tesseract_training_data.py <scan.png> [output_dir]
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def find_all_stamp_areas(image_path, debug=False):
    """
    Find all stamp areas on a page by detecting red regions.
    Each red date cluster = one stamp.
    Returns list of (x, y, w, h) bounding boxes.
    """
    # Handle PDF files
    if image_path.lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(image_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    else:
        img = Image.open(image_path)
    
    arr = np.array(img)
    
    if len(arr.shape) < 3:
        print("Error: Image must be RGB")
        return []
    
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    
    # Find red pixels (R much higher than G and B)
    red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter: must be reasonably large (date characters)
    h_img, w_img = red_mask.shape
    min_area = h_img * w_img * 0.0001  # 0.01% of page
    
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Must be in reasonable position (not at very bottom)
        if y > h_img * 0.9:
            continue
        # Must be wider than tall (date text)
        if h > 0 and w / h < 0.5:
            continue
        valid_contours.append((x, y, w, h, area))
    
    if not valid_contours:
        print("No red regions found!")
        return []
    
    # Group nearby contours that belong to the same stamp
    groups = []
    used = set()
    
    for i, (x1, y1, w1, h1, a1) in enumerate(valid_contours):
        if i in used:
            continue
        
        group = [(x1, y1, w1, h1)]
        used.add(i)
        
        for j, (x2, y2, w2, h2, a2) in enumerate(valid_contours):
            if j in used or i == j:
                continue
            
            date_h = h1
            if abs(y1 - y2) < date_h * 2 and abs(x1 - x2) < date_h * 3:
                group.append((x2, y2, w2, h2))
                used.add(j)
        
        groups.append(group)
    
    # Convert groups to stamp bounding boxes
    stamp_boxes = []
    for group in groups:
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[2] for b in group]
        hs = [b[3] for b in group]
        
        date_h = max(y + h for y, h in zip(ys, hs)) - min(ys)
        
        # Expand to full stamp area
        min_x = max(0, min(xs) - int(date_h * 2))
        min_y = max(0, min(ys) - int(date_h * 3))
        max_x = min(w_img, max(x + w for x, w in zip(xs, ws)) + int(date_h * 2))
        max_y = min(h_img, max(y + h for y, h in zip(ys, hs)) + int(date_h * 2))
        
        stamp_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    return stamp_boxes


def auto_read_date(stamp_img):
    """
    Try to auto-read the date from the red channel.
    Returns the date string or empty string if failed.
    """
    try:
        import pytesseract
        arr = np.array(stamp_img)
        if len(arr.shape) < 3:
            return ""
        
        r = arr[:,:,0].astype(int)
        g = arr[:,:,1].astype(int)
        b = arr[:,:,2].astype(int)
        
        red_only = np.zeros_like(arr)
        red_mask = (r > g + 15) & (r > b + 15) & (r > 80)
        red_only[red_mask] = [255, 255, 255]
        
        from PIL import ImageChops
        red_img = Image.fromarray(red_only.astype(np.uint8))
        red_inv = ImageChops.invert(red_img.convert("RGB"))
        
        red_inv = red_inv.convert("L")
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(red_inv)
        red_inv = enhancer.enhance(2.5)
        red_inv = red_inv.point(lambda x: 0 if x < 100 else 255, mode="1")
        red_inv = red_inv.resize((red_inv.width * 4, red_inv.height * 4), Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(red_inv, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ').upper().strip()
        return text
    except Exception:
        return ""


def generate_training_data(scan_path, output_dir=None):
    """
    Main entry point: process a multi-stamp scan into training data.
    Handles multi-page PDFs.
    """
    scan_path = Path(scan_path)
    if output_dir is None:
        output_dir = scan_path.parent / "training_output"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing scan: {scan_path}")
    
    # Load all pages
    if str(scan_path).lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(scan_path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    else:
        pages = [Image.open(scan_path)]
    
    print(f"  {len(pages)} page(s)")
    
    total_stamps = 0
    
    for page_num, img in enumerate(pages, 1):
        # Find all stamps on this page
        # Need to save temp image for find_all_stamp_areas
        temp_path = output_dir / f"_temp_page_{page_num}.png"
        img.save(temp_path)
        stamp_boxes = find_all_stamp_areas(str(temp_path))
        temp_path.unlink()
        
        if not stamp_boxes:
            print(f"  Page {page_num}: no stamps found")
            continue
        
        print(f"  Page {page_num}: {len(stamp_boxes)} stamp(s)")
        
        # Extract each stamp
        for i, box in enumerate(stamp_boxes):
            x, y, w, h = box
            
            stamp_img = img.crop((x, y, x + w, y + h))
            
            stamp_name = f"stamp_{total_stamps+1:03d}"
            stamp_img.save(output_dir / f"{stamp_name}.png")
            
            # Try to auto-read the date
            date_text = auto_read_date(stamp_img)
            
            # Generate ground truth text file
            gt_path = output_dir / f"{stamp_name}.gt.txt"
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(f"RECEIVED\n{date_text}\nBY:")
            
            print(f"    {stamp_name}.png ({w}x{h}) - date: '{date_text}'")
            total_stamps += 1
    
    if total_stamps == 0:
        print("No stamps detected!")
        return
    
    # Create a list of all training files
    with open(output_dir / "training_list.txt", 'w') as f:
        for i in range(total_stamps):
            f.write(f"stamp_{i+1:03d}\n")
    
    print(f"\nTraining data written to: {output_dir}")
    print(f"  {total_stamps} stamp images")
    print(f"  {total_stamps} ground truth files")
    print(f"\nNext steps:")
    print(f"  1. Review and correct dates in the .gt.txt files")
    print(f"  2. Run: cd {output_dir}")
    print(f"  3. Run: make training MODEL_NAME=stamp START_MODEL=eng")
    print(f"\nThen deploy:")
    print(f"  copy stamp.traineddata \"C:\\Program Files\\Tesseract-OCR\\tessdata\\\"")
    print(f"\nUsage in code:")
    print(f"  pytesseract.image_to_string(img, config='--psm 6 -l eng+stamp')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    scan_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(scan_path).exists():
        print(f"File not found: {scan_path}")
        sys.exit(1)
    
    generate_training_data(scan_path, output_dir)
