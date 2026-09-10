#!/usr/bin/env python3
"""
tesseract_train.py

Complete Tesseract training pipeline for the RECEIVED stamp font.

Steps:
  1. Extract stamps from scan at maximum quality
  2. Generate initial ground truth using best-effort OCR
  3. Create box files for Tesseract
  4. Run training
  5. Deploy the trained model

Usage:
  python tesseract_train.py <scan.pdf> [output_dir]
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

def extract_stamps(scan_path, output_dir):
    """Extract all stamps from a scan."""
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


def generate_ground_truth(stamp_img, debug=False):
    """
    Generate ground truth text for a stamp.
    Uses multiple OCR approaches and returns the best result.
    """
    import pytesseract
    
    # Upscale significantly
    large = stamp_img.resize((stamp_img.width * 6, stamp_img.height * 6), Image.Resampling.LANCZOS)
    
    results = []
    
    # Method 1: Direct OCR on RGB
    for psm in [6, 7, 4]:
        try:
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            result = pytesseract.image_to_string(large, config=config).upper().strip()
            if result:
                results.append(result)
        except:
            pass
    
    # Method 2: Grayscale + contrast + sharpen
    gray = large.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.SHARPEN)
    for psm in [6, 7]:
        try:
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            result = pytesseract.image_to_string(enhanced, config=config).upper().strip()
            if result:
                results.append(result)
        except:
            pass
    
    # Method 3: Otsu threshold
    arr = np.array(gray)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = Image.fromarray(binary)
    for psm in [6, 7]:
        try:
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
            result = pytesseract.image_to_string(binary_img, config=config).upper().strip()
            if result:
                results.append(result)
        except:
            pass
    
    # Return the longest result (likely the most complete)
    if results:
        return max(results, key=len)
    return ""


def create_box_file(image, ground_truth, output_path):
    """
    Create a Tesseract box file from an image and its ground truth text.
    
    Tesseract box format: char left bottom right top page
    (bottom-left origin, so we need to flip y coordinates)
    """
    img_w, img_h = image.size
    
    lines = ground_truth.strip().split('\n')
    
    # Divide image into 3 vertical zones for 3 lines of text
    line_height = img_h // 3
    
    boxes = []
    
    for line_idx, line in enumerate(lines[:3]):
        if not line.strip():
            continue
        
        y_top = line_idx * line_height
        y_bottom = (line_idx + 1) * line_height
        
        # Split line into characters
        chars = list(line.strip())
        if not chars:
            continue
        
        char_width = img_w // len(chars)
        
        for char_idx, char in enumerate(chars):
            left = char_idx * char_width
            right = (char_idx + 1) * char_width
            
            # Tesseract box format: char left bottom right top page
            # bottom-left origin, so y is flipped
            boxes.append((char, left, img_h - y_bottom, right, img_h - y_top, 0))
    
    # Write box file
    with open(output_path, 'w', encoding='utf-8') as f:
        for char, left, bottom, right, top, page in boxes:
            f.write(f'{char} {left} {bottom} {right} {top} {page}\n')
    
    return boxes


def train_tesseract(training_dir, model_name='stamp'):
    """
    Run Tesseract training using tesstrain.
    """
    import subprocess
    
    training_dir = Path(training_dir)
    
    # Create training data directory structure
    train_dir = training_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    # Copy files to training directory
    for gt_file in training_dir.glob("*.gt.txt"):
        stem = gt_file.stem.replace('.gt', '')
        
        # Copy image
        img_file = training_dir / f"{stem}.png"
        if img_file.exists():
            # Create high-res version for training
            img = Image.open(img_file)
            target_w = max(img.width * 4, 400)
            target_h = max(img.height * 4, 400)
            large = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            large.save(train_dir / f"{stem}.png")
            
            # Copy box file
            box_file = training_dir / f"{stem}.box"
            if box_file.exists():
                import shutil
                shutil.copy(box_file, train_dir / f"{stem}.box")
            
            # Copy ground truth
            import shutil
            shutil.copy(gt_file, train_dir / f"{stem}.gt.txt")
    
    # Run tesstrain
    cmd = [
        'make', 'training',
        f'MODEL_NAME={model_name}',
        'START_MODEL=eng',
        f'DATA_DIR={train_dir}',
        'MAX_ITERATIONS=1000'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Training completed successfully!")
        print(f"Model saved to: {train_dir}/{model_name}.traineddata")
    else:
        print("Training failed!")
        print(result.stdout)
        print(result.stderr)
    
    return result.returncode == 0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    scan_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("stamp_training")
    
    if not scan_path.exists():
        print(f"File not found: {scan_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing: {scan_path}")
    
    # Step 1: Extract stamps
    stamps = extract_stamps(str(scan_path), output_dir)
    print(f"Found {len(stamps)} stamps")
    
    if not stamps:
        print("No stamps detected!")
        sys.exit(1)
    
    # Step 2: Generate ground truth and box files
    for i, stamp in enumerate(stamps):
        stamp_name = f"stamp_{i+1:03d}"
        img = stamp['image']
        
        # Save stamp image
        img.save(output_dir / f"{stamp_name}.png")
        
        # Generate ground truth
        gt_text = generate_ground_truth(img)
        
        # Write ground truth file
        with open(output_dir / f"{stamp_name}.gt.txt", 'w', encoding='utf-8') as f:
            f.write(f"RECEIVED\n{gt_text}\nBY:")
        
        # Create box file
        create_box_file(img, f"RECEIVED\n{gt_text}\nBY:", output_dir / f"{stamp_name}.box")
        
        print(f"  {stamp_name}: {img.size} - gt: '{gt_text[:30]}'")
    
    # Step 3: Create training list
    with open(output_dir / "training_list.txt", 'w') as f:
        for i in range(len(stamps)):
            f.write(f"stamp_{i+1:03d}\n")
    
    print(f"\nTraining data ready in: {output_dir}")
    print(f"  {len(stamps)} stamps")
    print(f"\nTo train Tesseract:")
    print(f"  cd {output_dir}")
    print(f"  python tesseract_train.py {scan_path} {output_dir}")
    print(f"\nOr manually:")
    print(f"  make training MODEL_NAME=stamp START_MODEL=eng DATA_DIR={output_dir}/train")


if __name__ == "__main__":
    main()
