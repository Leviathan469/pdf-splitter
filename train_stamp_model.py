#!/usr/bin/env python3
"""
train_stamp_model.py

Train Tesseract on existing stamp images using the raw Tesseract training pipeline.

Prerequisites:
  - Tesseract installed with training tools
  - Training images as PNG files
  - Ground truth .gt.txt files
  - Box files (.box)

Usage:
  python train_stamp_model.py <training_dir> [model_name]
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image

def run_cmd(cmd, cwd=None):
    """Run a shell command and return success status."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
    return result.returncode == 0

def train_model(training_dir, model_name='stamp'):
    """
    Train Tesseract model from existing images and box files.
    """
    training_dir = Path(training_dir)
    
    if not training_dir.exists():
        print(f"Directory not found: {training_dir}")
        return False
    
    # Find all training images
    images = list(training_dir.glob("*.png"))
    if not images:
        print("No PNG images found!")
        return False
    
    print(f"Found {len(images)} training images")
    
    # Create TIFF versions (required for training)
    for img_path in images:
        tiff_path = img_path.with_suffix('.tif')
        if not tiff_path.exists():
            img = Image.open(img_path)
            img.save(tiff_path, compression='none')
    
    # Check for box files
    box_files = list(training_dir.glob("*.box"))
    if not box_files:
        print("No box files found! Generate them first.")
        return False
    
    print(f"Found {len(box_files)} box files")
    
    # Check for ground truth files
    gt_files = list(training_dir.glob("*.gt.txt"))
    if not gt_files:
        print("No ground truth files found!")
        return False
    
    print(f"Found {len(gt_files)} ground truth files")
    
    # Step 1: Generate .tr files (training data)
    print("\nStep 1: Generating training data files...")
    for img_path in images:
        tiff_path = img_path.with_suffix('.tif')
        box_path = img_path.with_suffix('.box')
        tr_path = img_path.with_suffix('.tr')
        
        if not box_path.exists():
            print(f"  Skipping {img_path.name} (no box file)")
            continue
        
        # Generate .tr file from image and box
        cmd = [
            'tesseract', str(tiff_path), str(tr_path.with_suffix('')),
            'box.train', 'stderr'
        ]
        if not run_cmd(cmd):
            print(f"  Failed to generate {tr_path.name}")
            continue
    
    # Step 2: Extract unicharset
    print("\nStep 2: Extracting unicharset...")
    box_files_str = ' '.join(str(b) for b in training_dir.glob("*.box"))
    cmd = f"unicharset_extractor {' '.join(str(b) for b in training_dir.glob('*.box'))}"
    if not run_cmd(cmd.split()):
        print("  Failed to extract unicharset")
        return False
    
    # Step 3: Create font_properties file
    print("\nStep 3: Creating font_properties...")
    font_props = training_dir / "font_properties"
    with open(font_props, 'w') as f:
        f.write("stamp 0 0 0 0 0\n")
    
    # Step 4: Run mftraining
    print("\nStep 4: Running mftraining...")
    unicharset = training_dir / "unicharset"
    cmd = [
        'mftraining',
        '-F', str(font_props),
        '-U', str(unicharset),
        '-O', f'{model_name}.unicharset'
    ] + [str(tr) for tr in training_dir.glob("*.tr")]
    
    if not run_cmd(cmd):
        print("  mftraining failed")
        return False
    
    # Step 5: Run cntraining
    print("\nStep 5: Running cntraining...")
    cmd = ['cntraining'] + [str(tr) for tr in training_dir.glob("*.tr")]
    if not run_cmd(cmd):
        print("  cntraining failed")
        return False
    
    # Step 6: Combine into traineddata
    print("\nStep 6: Combining traineddata...")
    cmd = [
        'combine_tessdata', f'{model_name}.'
    ]
    if not run_cmd(cmd):
        print("  combine_tessdata failed")
        return False
    
    # Step 7: Copy to tessdata
    traineddata = training_dir / f"{model_name}.traineddata"
    if traineddata.exists():
        tessdata_dir = Path(r"C:\Program Files\Tesseract-OCR\tessdata")
        if tessdata_dir.exists():
            import shutil
            shutil.copy(traineddata, tessdata_dir / f"{model_name}.traineddata")
            print(f"\nModel deployed to: {tessdata_dir / f'{model_name}.traineddata'}")
        else:
            print(f"\nModel saved to: {traineddata}")
            print(f"Copy to tessdata manually: copy {traineddata} \"C:\\Program Files\\Tesseract-OCR\\tessdata\\\"")
    
    print("\nTraining complete!")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    training_dir = Path(sys.argv[1])
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'stamp'
    
    # Add Tesseract to PATH
    tesseract_dir = r"C:\Program Files\Tesseract-OCR"
    if tesseract_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{tesseract_dir};{os.environ.get('PATH', '')}"
    
    success = train_model(training_dir, model_name)
    sys.exit(0 if success else 1)
