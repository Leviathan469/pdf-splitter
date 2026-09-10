#!/usr/bin/env python3
"""
train_on_work_pc.py

Complete Tesseract training pipeline for the RECEIVED stamp font.
Run this on your WORK PC where Tesseract is fully installed.

Prerequisites:
  - Tesseract installed (v5.x) with training tools
  - Python 3.8+ with packages: Pillow, numpy, opencv-python, pytesseract, tesstrain
  - A multi-stamp scan (PDF or PNG)
  - Writing permissions to tessdata directory

Usage:
  python train_on_work_pc.py <scan.pdf> [model_name]
  
Workflow:
  1. Extract stamps from scan using red channel detection
  2. Generate ground truth files (auto-read + manual verification)
  3. Generate box files for Tesseract training
  4. Run mftraining + cntraining (classifier training)
  5. Run lstmtraining (fine-tune existing eng model) — OPTIONAL BUT RECOMMENDED
  6. Combine and deploy traineddata

For the stamp font (outlined/hollow), we use BOTH:
  - Traditional classifier training (for character shape recognition)
  - LSTM fine-tuning (for context-aware reading)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# ============================================================================
# CONFIGURATION — CHANGE THESE TO MATCH YOUR WORK PC
# ============================================================================
TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"  # Change if different
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")
TESSERACT_EXE = os.path.join(TESSERACT_DIR, "tesseract.exe")

# Where to put the training data
OUTPUT_DIR = Path("stamp_training_output")

# Model name (the trained file will be named <MODEL_NAME>.traineddata)
MODEL_NAME = "stamp"

# Training text for LSTM (vocabulary the stamp uses)
TRAINING_TEXT = """
RECEIVED JAN 01 2026 BY:
RECEIVED FEB 02 2026 BY:
RECEIVED MAR 13 2026 BY:
RECEIVED APR 14 2026 BY:
RECEIVED MAY 25 2026 BY:
RECEIVED JUN 25 2026 BY:
RECEIVED JUL 16 2026 BY:
RECEIVED AUG 31 2026 BY:
RECEIVED SEP 27 2026 BY:
RECEIVED OCT 28 2026 BY:
RECEIVED NOV 29 2026 BY:
RECEIVED DEC 30 2026 BY:
RECEIVED SEP 10 2027 BY:
"""

# ============================================================================
# STEP 1: EXTRACT STAMPS FROM SCAN
# ============================================================================

def find_stamp_areas(image_path):
    """Find all stamp areas on a page by detecting red regions."""
    if str(image_path).lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(image_path)
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    else:
        pages = [Image.open(image_path)]
    
    all_stamps = []
    
    for page_num, img in enumerate(pages, 1):
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
                'bbox': (min_x, min_y, max_x - min_x, max_y - min_y)
            })
    
    return all_stamps


# ============================================================================
# STEP 2: GENERATE GROUND TRUTH AND BOX FILES
# ============================================================================

def auto_read_stamp(stamp_img):
    """Auto-read stamp text using multiple OCR approaches."""
    try:
        import pytesseract
        large = stamp_img.resize((stamp_img.width * 6, stamp_img.height * 6), Image.Resampling.LANCZOS)
        
        results = []
        for psm in [6, 7, 4]:
            try:
                config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
                result = pytesseract.image_to_string(large, config=config).upper().strip()
                if result:
                    results.append(result)
            except:
                pass
        
        if results:
            return max(results, key=len)
    except:
        pass
    return ""


def generate_box_file(image, ground_truth, output_path):
    """Generate Tesseract box file."""
    img_w, img_h = image.size
    lines = ground_truth.strip().split('\n')
    
    zones = [
        (0, int(img_h * 0.40)),
        (int(img_h * 0.35), int(img_h * 0.70)),
        (int(img_h * 0.65), img_h)
    ]
    
    boxes = []
    for line_idx, (line, (zone_top, zone_bottom)) in enumerate(zip(lines, zones)):
        if not line:
            continue
        chars = list(line)
        char_width = img_w / len(chars)
        for char_idx, char in enumerate(chars):
            left = int(char_idx * char_width)
            right = int((char_idx + 1) * char_width)
            boxes.append((char, left, img_h - zone_bottom, right, img_h - zone_top, 0))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for char, left, bottom, right, top, page in boxes:
            f.write(f'{char} {left} {bottom} {right} {top} {page}\n')
    
    return boxes


# ============================================================================
# STEP 3: TRAINING
# ============================================================================

def run_cmd(cmd, cwd=None):
    """Run a command."""
    import subprocess
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(f"  STDOUT: {result.stdout[:500]}")
    if result.stderr:
        print(f"  STDERR: {result.stderr[:500]}")
    return result.returncode == 0


def train_traditional(training_dir, model_name):
    """Traditional classifier training (mftraining + cntraining)."""
    training_dir = Path(training_dir)
    
    # Check for training files
    tr_files = sorted(training_dir.glob("*.tr"))
    if not tr_files:
        print("No .tr files found!")
        return False
    
    print(f"\n=== Training with {len(tr_files)} files ===")
    
    # Step 1: Extract unicharset
    print("\nStep 1: Extracting unicharset...")
    box_files = sorted(training_dir.glob("*.box"))
    cmd = ['unicharset_extractor'] + [str(b) for b in box_files]
    if not run_cmd(cmd):
        print("unicharset_extractor failed")
    
    # Step 2: Create font_properties
    print("\nStep 2: Creating font_properties...")
    font_props = training_dir / "font_properties"
    with open(font_props, 'w') as f:
        f.write(f"{model_name} 0 0 0 0 0\n")
    
    # Step 3: mftraining
    print("\nStep 3: Running mftraining...")
    cmd = ['mftraining', '-F', str(font_props), '-U', str(training_dir / "unicharset"), 
           '-O', f'{model_name}.unicharset'] + [str(tr) for tr in tr_files]
    if not run_cmd(cmd):
        print("mftraining failed")
    
    # Step 4: cntraining
    print("\nStep 4: Running cntraining...")
    cmd = ['cntraining'] + [str(tr) for tr in tr_files]
    if not run_cmd(cmd):
        print("cntraining failed")
    
    # Step 5: Rename files
    print("\nStep 5: Renaming files...")
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
    
    # Step 6: Combine
    print("\nStep 6: Combining traineddata...")
    cmd = ['combine_tessdata', f'{model_name}.']
    if run_cmd(cmd, cwd=training_dir):
        traineddata = training_dir / f"{model_name}.traineddata"
        if traineddata.exists():
            print(f"\n✅ Model created: {traineddata}")
            
            # Deploy to tessdata
            dest = Path(TESSDATA_DIR) / f"{model_name}.traineddata"
            try:
                import shutil
                shutil.copy(traineddata, dest)
                print(f"Deployed to: {dest}")
            except Exception as e:
                print(f"Could not deploy (need admin?): {e}")
                print(f"Manually copy: {traineddata} -> {dest}")
            
            return True
    
    print("Training failed!")
    return False


def train_lstm(training_dir, model_name):
    """
    LSTM fine-tuning (requires existing traineddata to start from).
    This is the BEST approach for Tesseract v5.
    """
    training_dir = Path(training_dir)
    
    print("\n=== LSTM Fine-tuning ===")
    
    # Extract eng components
    eng_traineddata = Path(TESSDATA_DIR) / "eng.traineddata"
    if not eng_traineddata.exists():
        print("eng.traineddata not found! Cannot do LSTM fine-tuning.")
        return False
    
    # Extract eng components
    cmd = ['combine_tessdata', '-e', str(eng_traineddata), 'eng.']
    if not run_cmd(cmd, cwd=training_dir):
        print("Failed to extract eng components")
        return False
    
    # Create training text file
    training_text_file = training_dir / "training_text.txt"
    with open(training_text_file, 'w') as f:
        f.write(TRAINING_TEXT)
    
    # Generate LSTM training data from box files
    box_files = sorted(training_dir.glob("*.box"))
    if not box_files:
        print("No box files found!")
        return False
    
    # Generate .lstmf files
    for box_file in box_files:
        tiff_file = box_file.with_suffix('.tif')
        if not tiff_file.exists():
            png_file = box_file.with_suffix('.png')
            if png_file.exists():
                Image.open(png_file).save(tiff_file, compression='none')
        
        if tiff_file.exists():
            lstmf_file = box_file.with_suffix('.lstmf')
            cmd = ['tesseract', str(tiff_file), str(lstmf_file.with_suffix('')), 'lstm.train']
            run_cmd(cmd)
    
    # Create list of lstmf files
    lstmf_files = sorted(training_dir.glob("*.lstmf"))
    if not lstmf_files:
        print("No .lstmf files generated!")
        return False
    
    lstmf_list = training_dir / "training_list.txt"
    with open(lstmf_list, 'w') as f:
        for lf in lstmf_files:
            f.write(f"{lf}\n")
    
    # Run lstmtraining
    print("\nRunning lstmtraining...")
    cmd = [
        'lstmtraining',
        '--model_output', f'{model_name}',
        '--continue_from', str(training_dir / "eng.lstm"),
        '--traineddata', str(eng_traineddata),
        '--train_listfile', str(lstmf_list),
        '--max_iterations', '1000'
    ]
    
    if run_cmd(cmd, cwd=training_dir):
        # Combine with eng
        cmd = ['combine_tessdata', f'{model_name}.']
        if run_cmd(cmd, cwd=training_dir):
            traineddata = training_dir / f"{model_name}.traineddata"
            if traineddata.exists():
                print(f"\n✅ LSTM model created: {traineddata}")
                return True
    
    return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python train_on_work_pc.py <scan.pdf> [model_name]")
        sys.exit(1)
    
    scan_path = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    
    if not scan_path.exists():
        print(f"File not found: {scan_path}")
        sys.exit(1)
    
    # Setup
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add tesseract to PATH
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    
    print(f"Scan: {scan_path}")
    print(f"Output: {output_dir}")
    print(f"Model name: {model_name}")
    
    # Step 1: Extract stamps
    print("\n" + "="*60)
    print("STEP 1: Extracting stamps from scan")
    print("="*60)
    stamps = find_stamp_areas(str(scan_path))
    print(f"Found {len(stamps)} stamps")
    
    if not stamps:
        print("No stamps detected! Check the scan has visible red ink.")
        sys.exit(1)
    
    # Step 2: Generate ground truth and box files
    print("\n" + "="*60)
    print("STEP 2: Generating ground truth and box files")
    print("="*60)
    
    for i, stamp in enumerate(stamps):
        stamp_name = f"stamp_{i+1:03d}"
        img = stamp['image']
        
        # Save stamp image
        img.save(output_dir / f"{stamp_name}.png")
        
        # Auto-read
        gt_text = auto_read_stamp(img)
        
        # Write ground truth
        with open(output_dir / f"{stamp_name}.gt.txt", 'w', encoding='utf-8') as f:
            f.write(gt_text)
        
        # Generate box file
        generate_box_file(img, gt_text, output_dir / f"{stamp_name}.box")
        
        print(f"  {stamp_name}: {img.size} - gt: '{gt_text[:50]}'")
    
    # Create training list
    with open(output_dir / "training_list.txt", 'w') as f:
        for i in range(len(stamps)):
            f.write(f"stamp_{i+1:03d}\n")
    
    print(f"\nTraining data ready in: {output_dir}")
    print(f"  {len(stamps)} stamp images")
    print(f"  {len(stamps)} ground truth files")
    print(f"  {len(stamps)} box files")
    
    # Step 3: Verify ground truth
    print("\n" + "="*60)
    print("STEP 3: Verify ground truth")
    print("="*60)
    print("Please review the .gt.txt files in the output directory")
    print("and correct any OCR errors before training.")
    
    response = input("\nContinue with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled. Fix ground truth files and re-run.")
        sys.exit(0)
    
    # Step 4: Train
    print("\n" + "="*60)
    print("STEP 4: Training")
    print("="*60)
    
    # Try traditional training first
    success = train_traditional(output_dir, model_name)
    
    if success:
        print("\n✅ Training complete!")
        print(f"\nUsage in code:")
        print(f"  pytesseract.image_to_string(img, config='--psm 6 -l {model_name}')")
    else:
        print("\n❌ Training failed. Check the error messages above.")


if __name__ == "__main__":
    main()
