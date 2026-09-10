import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def generate_box_files(training_dir):
    """Generate proper box files for Tesseract LSTM training."""
    training_dir = Path(training_dir)
    
    # Load the original scan to get full-res images
    import fitz
    doc = fitz.open(r'C:\Users\aiden\Downloads\Stamp training data.pdf')
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    original = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    
    # Find all stamp areas
    arr = np.array(original)
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    
    red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
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
    
    print(f"Found {len(groups)} stamps")
    
    # Dates for each row
    dates = [
        'JAN 01 2026', 'FEB 02 2026', 'MAR 13 2026',
        'APR 14 2026', 'MAY 25 2026', 'JUN 25 2026',
        'JUL 16 2026', 'AUG 31 2026', 'SEP 27 2026',
        'OCT 28 2026', 'NOV 29 2026', 'DEC 30 2026',
    ] + ['SEP 10 2027'] * 15  # Remaining are SEP 10 2027
    
    # For each group, extract stamp and generate box file
    for idx, (group, date) in enumerate(zip(groups, dates)):
        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        ws = [b[2] for b in group]
        hs = [b[3] for b in group]
        
        date_h = max(y+h for y, h in zip(ys, hs)) - min(ys)
        
        min_x = max(0, min(xs) - int(date_h * 2))
        min_y = max(0, min(ys) - int(date_h * 3))
        max_x = min(w_img, max(x+w for x, w in zip(xs, ws)) + int(date_h * 2))
        max_y = min(h_img, max(y+h for y, h in zip(ys, hs)) + int(date_h * 2))
        
        # Save stamp image at full resolution
        stamp_img = original.crop((min_x, min_y, max_x, max_y))
        stamp_path = training_dir / f"stamp_{idx+1:03d}.png"
        stamp_path_full = training_dir / f"stamp_{idx+1:03d}_full.png"
        stamp_img.save(stamp_path_full)
        
        # Also save at training size
        stamp_img.save(stamp_path)
        
        # Save TIFF for Tesseract
        tiff_path = training_dir / f"stamp_{idx+1:03d}.tif"
        stamp_img.save(tiff_path, compression='none')
        
        # Generate box file (3 lines: RECEIVED, date, BY:)
        # Box format: text left bottom right top page (bottom-left origin)
        box_path = training_dir / f"stamp_{idx+1:03d}.box"
        img_w, img_h = stamp_img.size
        
        # Divide into 3 zones
        zone_h = img_h // 3
        
        with open(box_path, 'w', encoding='utf-8') as f:
            # RECEIVED (top zone)
            f.write(f'RECEIVED 0 {img_h - zone_h} {img_w} {img_h} 0\n')
            # DATE (middle zone)
            f.write(f'{date} 0 {img_h - 2*zone_h} {img_w} {img_h - zone_h} 0\n')
            # BY: (bottom zone)
            f.write(f'BY: 0 0 {img_w} {img_h - 2*zone_h} 0\n')
        
        # Save ground truth
        gt_path = training_dir / f"stamp_{idx+1:03d}.gt.txt"
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(f'RECEIVED\n{date}\nBY:')
        
        print(f"  stamp_{idx+1:03d}: {stamp_img.size} - {date}")
    
    return len(groups)


if __name__ == "__main__":
    os.chdir(r'C:\Users\aiden\Documents\pdf-splitter')
    os.makedirs('training_output_clean', exist_ok=True)
    count = generate_box_files('training_output_clean')
    print(f"\nGenerated {count} stamp training sets")
