#!/usr/bin/env python3
"""
train_with_synthetic_data.py

Generate synthetic training data using text2image with a font similar to the stamp font,
then train Tesseract on that data.

The stamp font appears to be a condensed slab serif. We'll use the closest available
font on Windows and generate synthetic images with various degradations.

Usage:
  python train_with_synthetic_data.py [font_name]
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")

def list_fonts():
    """List available fonts that might match the stamp font."""
    cmd = ['text2image', '--list_available_fonts', '--fonts_dir', 'C:/Windows/Fonts']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Look for condensed, slab, or impact fonts
    keywords = ['condensed', 'impact', 'slab', 'agency', 'bahnschrift', 'stencil']
    
    fonts = []
    for line in result.stdout.split('\n'):
        line_lower = line.lower().strip()
        for kw in keywords:
            if kw in line_lower:
                # Extract font name
                parts = line.split(':')
                if len(parts) > 1:
                    fonts.append((parts[0].strip(), parts[1].strip()))
    
    return fonts

def generate_training_data(font_name, output_dir, num_samples=100):
    """Generate synthetic training images with the given font."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    
    # Training text - variations of stamp text
    texts = []
    
    # RECEIVED variations
    for month in ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                  'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']:
        for day in ['01', '02', '03', '04', '05', '10', '13', '14', '15', '16', 
                    '25', '27', '28', '29', '30', '31']:
            for year in ['2026', '2027']:
                texts.append(f"RECEIVED\n{month} {day} {year}\nBY:")
    
    print(f"Generated {len(texts)} training texts")
    
    for i, text in enumerate(texts[:num_samples]):
        # Create temp file for text
        txt_file = output_dir / f"sample_{i:04d}.gt.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Generate image with text2image
        output_base = output_dir / f"sample_{i:04d}"
        
        cmd = [
            'text2image',
            '--font', font_name,
            '--text', str(txt_file),
            '--outputbase', str(output_base),
            '--ptsize', '30',
            '--xsize', '400',
            '--ysize', '100',
            '--leading', '20',
            '--char_spacing', '0.05',
            '--exposure', '0',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Try without --char_spacing
            cmd = [
                'text2image',
                '--font', font_name,
                '--text', str(txt_file),
                '--outputbase', str(output_base),
                '--ptsize', '30',
                '--xsize', '400',
                '--ysize', '100',
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            tif_file = output_base.with_suffix('.tif')
            if tif_file.exists():
                # Generate box file
                generate_box_file(text, tif_file, output_base.with_suffix('.box'))
                print(f"  OK: sample_{i:04d}")
            else:
                print(f"  FAIL: sample_{i:04d}")
        else:
            print(f"  ERROR: sample_{i:04d} - {result.stderr[:100]}")

def generate_box_file(text, tif_file, box_file):
    """Generate a box file for a given text."""
    from PIL import Image
    
    img = Image.open(tif_file)
    w, h = img.size
    
    lines = text.split('\n')
    zone_height = h // len(lines)
    
    boxes = []
    for line_idx, line in enumerate(lines):
        y_top = (len(lines) - line_idx - 1) * zone_height
        y_bottom = (len(lines) - line_idx) * zone_height
        
        # Estimate character positions (text2image centers text)
        char_width = w / max(len(line), 1)
        x_offset = (w - len(line) * char_width) / 2
        
        for char_idx, char in enumerate(line):
            left = int(x_offset + char_idx * char_width)
            right = int(x_offset + (char_idx + 1) * char_width)
            boxes.append(f'{char} {left} {y_top} {right} {y_bottom} 0')
    
    with open(box_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(boxes) + '\n')

def main():
    if len(sys.argv) < 2:
        print("Available fonts:")
        for idx, name in list_fonts():
            print(f"  {idx}: {name}")
        print(f"\nUsage: python {sys.argv[0]} <font_name>")
        sys.exit(0)
    
    font_name = sys.argv[1]
    output_dir = "synthetic_training_data"
    
    print(f"Using font: {font_name}")
    print(f"Output: {output_dir}")
    
    generate_training_data(font_name, output_dir, num_samples=50)

if __name__ == "__main__":
    main()
