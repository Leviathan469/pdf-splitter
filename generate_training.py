#!/usr/bin/env python3
"""
generate_training.py

Generate clean Tesseract training data from stamp images.
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")

def generate_box_file(img_w, img_h, lines, output_path):
    """
    Generate a proper Tesseract box file.
    Each line: char left bottom right top page
    One character per line, bottom-left origin.
    """
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


def main():
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    output_dir = Path('training_output_clean')
    output_dir.mkdir(exist_ok=True)
    
    import fitz
    doc = fitz.open(r'C:\Users\aiden\Downloads\Stamp training data.pdf')
    page = doc[0]
    pix = page.get_pixmap(dpi=300)
    original = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    print(f'Original: {original.size}')
    
    arr = np.array(original)
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
    red_mask = ((r > g + 15) & (r > b + 15) & (r > 80)).astype(np.uint8) * 255
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = red_mask.shape
    min_area = h_img * w_img * 0.0001
    valid = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area and cv2.boundingRect(c)[1] < h_img * 0.9]
    
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
    
    print(f'Found {len(groups)} stamps')
    
    dates = ['JAN 01 2026', 'FEB 02 2026', 'MAR 13 2026',
             'APR 14 2026', 'MAY 25 2026', 'JUN 25 2026',
             'JUL 16 2026', 'AUG 31 2026', 'SEP 27 2026',
             'OCT 28 2026', 'NOV 29 2026', 'DEC 30 2026'] + ['SEP 10 2027'] * 15
    
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
        
        stamp = original.crop((min_x, min_y, max_x, max_y))
        w, h = stamp.size
        
        stamp.save(output_dir / f'stamp_{idx+1:03d}.png')
        stamp.save(output_dir / f'stamp_{idx+1:03d}.tif', compression='none')
        
        lines = ['RECEIVED', date, 'BY:']
        with open(output_dir / f'stamp_{idx+1:03d}.gt.txt', 'w') as f:
            f.write('\n'.join(lines))
        
        generate_box_file(w, h, lines, output_dir / f'stamp_{idx+1:03d}.box')
        
        print(f'stamp_{idx+1:03d}: {w}x{h} - {date}')
    
    print(f'\n{len(groups)} stamps extracted to {output_dir}/')


if __name__ == "__main__":
    main()
