#!/usr/bin/env python3
"""
train_stamp_with_tesstrain.py

Train Tesseract on stamp data using the tesstrain wrapper.
tesstrain handles the full pipeline: box files -> .tr files -> unicharset -> mftraining -> cntraining -> combine

This produces a legacy classifier model that works with --oem 1 (Tesseract legacy engine).

Prerequisites:
  pip install tesstrain

Usage:
  python train_stamp_with_tesstrain.py <scan.pdf> [model_name]
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")

def extract_stamps(scan_path, output_dir):
    """Extract stamp images from a multi-stamp scan."""
    if scan_path.lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(scan_path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    else:
        pages = [Image.open(scan_path)]
    
    all_stamps = []
    
    for page_num, img in enumerate(pages, 1):
        import cv2
        import numpy as np
        
        arr = np.array(img)
        r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
        
        red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
        
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h_img, w_img = red_mask.shape
        min_area = h_img * w_img * 0.0001
        
        valid = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if y > h_img * 0.9:
                continue
            valid.append((x, y, w, h))
        
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
        
        for group in groups:
            xs = [b[0] for b in group]
            ys = [b[1] for b in group]
            ws = [b[2] for b in group]
            hs = [b[3] for b in group]
            
            date_h = max(y + h for y, h in zip(ys, hs)) - min(ys)
            
            min_x = max(0, min(xs) - int(date_h * 2))
            min_y = max(0, min(ys) - int(date_h * 3))
            max_x = min(w_img, max(x + w for x, w in zip(xs, ws)) + int(date_h * 2))
            max_y = min(h_img, max(y + h for y, h in zip(ys, hs)) + int(date_h * 2))
            
            stamp_img = img.crop((min_x, min_y, max_x, max_y))
            all_stamps.append({
                'image': stamp_img,
                'page': page_num,
            })
    
    return all_stamps


def generate_box_file(img_w, img_h, lines, output_path):
    """Generate a proper Tesseract box file."""
    zone_height = img_h // len(lines)
    
    boxes = []
    for line_idx, line in enumerate(lines):
        if not line:
            continue
        
        y_top = (len(lines) - line_idx - 1) * zone_height
        y_bottom = (len(lines) - line_idx) * zone_height
        
        chars = list(line)
        char_width = img_w / len(chars)
        
        for char_idx, char in enumerate(chars):
            left = int(char_idx * char_width)
            right = int((char_idx + 1) * char_width)
            boxes.append(f'{char} {left} {y_top} {right} {y_bottom} 0\n')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(boxes)
    
    return boxes


def create_training_data(stamps, output_dir):
    """Create box and ground truth files for each stamp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dates = ['JAN 01 2026', 'FEB 02 2026', 'MAR 13 2026',
             'APR 14 2026', 'MAY 25 2026', 'JUN 25 2026',
             'JUL 16 2026', 'AUG 31 2026', 'SEP 27 2026',
             'OCT 28 2026', 'NOV 29 2026', 'DEC 30 2026'] + ['SEP 10 2027'] * 15
    
    for idx, (stamp, date) in enumerate(zip(stamps, dates)):
        stamp_name = f"stamp_{idx+1:03d}"
        img = stamp['image']
        
        # Save stamp image (PNG for tesstrain)
        png_path = output_dir / f"{stamp_name}.png"
        img.save(png_path)
        
        # Save TIFF (required for training)
        tif_path = output_dir / f"{stamp_name}.tif"
        img.save(tif_path, compression='none')
        
        # Create ground truth
        lines = ['RECEIVED', date, 'BY:']
        gt_path = output_dir / f"{stamp_name}.gt.txt"
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # Create box file
        w, h = img.size
        box_path = output_dir / f"{stamp_name}.box"
        generate_box_file(w, h, lines, box_path)
        
        print(f"  {stamp_name}: {w}x{h} - {date}")
    
    return len(stamps)


def run_tesstrain(training_dir, model_name):
    """Run tesstrain training."""
    training_dir = Path(training_dir)
    
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    
    # Find all box files
    box_files = sorted(training_dir.glob("*.box"))
    
    if not box_files:
        print("No box files found!")
        return False
    
    print(f"Found {len(box_files)} box files")
    
    # Run unicharset_extractor
    print("\nStep 1: Extracting unicharset...")
    cmd = ['unicharset_extractor'] + [str(b) for b in box_files]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(training_dir))
    print(f"  Return code: {result.returncode}")
    
    # Create font_properties
    print("\nStep 2: Creating font_properties...")
    font_props = training_dir / "font_properties"
    with open(font_props, 'w') as f:
        f.write(f"{model_name} 0 0 0 0 0\n")
    
    # Generate .tr files
    print("\nStep 3: Generating .tr files...")
    for box_file in box_files:
        tif_file = box_file.with_suffix('.tif')
        tr_file = box_file.with_suffix('.tr')
        
        if not tif_file.exists():
            print(f"  SKIP: {box_file.name} (no tif)")
            continue
        
        cmd = ['tesseract', str(tif_file), str(tr_file.with_suffix('')), 'box.train', 'stderr']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(training_dir))
        if result.returncode == 0 and tr_file.exists():
            print(f"  OK: {tr_file.name}")
        else:
            print(f"  FAIL: {box_file.name}")
    
    tr_files = sorted(training_dir.glob("*.tr"))
    if not tr_files:
        print("No .tr files generated!")
        return False
    
    # Run mftraining
    print("\nStep 4: Running mftraining...")
    cmd = [
        'mftraining',
        '-F', str(font_props),
        '-U', str(training_dir / "unicharset"),
        '-O', f'{model_name}.unicharset'
    ] + [str(tr) for tr in tr_files]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(training_dir))
    print(f"  Return code: {result.returncode}")
    
    # Run cntraining
    print("\nStep 5: Running cntraining...")
    cmd = ['cntraining'] + [str(tr) for tr in tr_files]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(training_dir))
    print(f"  Return code: {result.returncode}")
    
    # Rename files for combining
    for src_name, dst_name in [
        ('unicharset', f'{model_name}.unicharset'),
        ('inttemp', f'{model_name}.inttemp'),
        ('shapetable', f'{model_name}.shapetable'),
        ('pffmtable', f'{model_name}.pffmtable')
    ]:
        src = training_dir / src_name
        dst = training_dir / dst_name
        if src.exists() and not dst.exists():
            src.rename(dst)
    
    # Combine into traineddata
    print("\nStep 6: Combining traineddata...")
    cmd = ['combine_tessdata', f'{model_name}.']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(training_dir))
    print(f"  Return code: {result.returncode}")
    if result.stdout:
        print(f"  STDOUT: {result.stdout}")
    if result.stderr:
        print(f"  STDERR: {result.stderr}")
    
    traineddata = training_dir / f"{model_name}.traineddata"
    if traineddata.exists():
        print(f"\n✅ Model created: {traineddata} ({traineddata.stat().st_size} bytes)")
        
        # Copy to tessdata (try user dir first)
        user_tessdata = Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata"
        user_tessdata.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(traineddata, user_tessdata / traineddata.name)
        print(f"Deployed to: {user_tessdata / traineddata.name}")
        
        return True
    else:
        print("No traineddata created!")
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python train_stamp_with_tesstrain.py <scan.pdf> [model_name]")
        sys.exit(1)
    
    scan_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else "stamp"
    
    if not scan_path.exists():
        print(f"File not found: {scan_path}")
        sys.exit(1)
    
    training_dir = Path("tesstrain_data")
    training_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Training Stamp Font with tesstrain")
    print("=" * 60)
    
    # Step 1: Extract stamps
    print("\nStep 1: Extracting stamps from scan...")
    stamps = extract_stamps(str(scan_path), training_dir)
    print(f"  Found {len(stamps)} stamps")
    
    if not stamps:
        print("No stamps detected!")
        sys.exit(1)
    
    # Step 2: Create training data
    print("\nStep 2: Creating training data...")
    count = create_training_data(stamps, training_dir)
    print(f"  Created {count} training samples")
    
    # Step 3: Run tesstrain
    print("\nStep 3: Training model...")
    success = run_tesstrain(training_dir, model_name)
    
    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nUsage:")
        print(f"  pytesseract.image_to_string(img, config='--psm 6 --oem 1 -l {model_name}')")
        print(f"\nOr with both eng and stamp:")
        print(f"  pytesseract.image_to_string(img, config='--psm 6 --oem 1 -l eng+{model_name}')")
    else:
        print("\nTraining failed.")


if __name__ == "__main__":
    main()
