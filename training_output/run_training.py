#!/usr/bin/env python3
"""
run_training.py

Execute Tesseract training pipeline using the command-line tools.
"""

import os
import sys
import subprocess
from pathlib import Path

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = r"C:\Program Files\Tesseract-OCR\tessdata"

def run_cmd(cmd, cwd=None):
    """Run a command and print output."""
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        print(f"  OUT: {result.stdout[:200]}")
    if result.stderr:
        print(f"  ERR: {result.stderr[:200]}")
    if result.returncode != 0:
        print(f"  FAILED with code {result.returncode}")
    return result.returncode == 0

def main():
    training_dir = Path.cwd()
    model_name = "stamp"
    
    # Add tesseract to PATH
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    
    # Also set TESSDATA_PREFIX for finding traineddata
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    
    print(f"Training directory: {training_dir}")
    print(f"Tesseract dir: {TESSERACT_DIR}")
    print(f"Tessdata dir: {TESSDATA_DIR}")
    
    # Get list of training images
    images = sorted(training_dir.glob("stamp_*.png"))
    print(f"\nFound {len(images)} training images")
    
    if not images:
        print("No training images found!")
        return False
    
    # Step 1: Convert PNG to TIFF (required for training)
    print("\n=== Step 1: Converting PNG to TIFF ===")
    for img in images:
        tiff = img.with_suffix('.tif')
        if not tiff.exists():
            # Use PIL to convert
            from PIL import Image
            Image.open(img).save(tiff, compression='none')
    
    tiffs = sorted(training_dir.glob("stamp_*.tif"))
    print(f"Created {len(tiffs)} TIFF files")
    
    # Step 2: Generate .tr files (training data)
    print("\n=== Step 2: Generating .tr files ===")
    for img in images:
        tiff = img.with_suffix('.tif')
        box = img.with_suffix('.box')
        tr = img.with_suffix('.tr')
        
        if not box.exists():
            print(f"  Skipping {img.name} (no box file)")
            continue
        
        if not tiff.exists():
            print(f"  Skipping {img.name} (no tiff file)")
            continue
        
        # Generate .tr file
        cmd = ['tesseract', str(tiff), str(tr.with_suffix('')), 'box.train', 'stderr']
        run_cmd(cmd)
    
    tr_files = sorted(training_dir.glob("stamp_*.tr"))
    print(f"Generated {len(tr_files)} .tr files")
    
    if not tr_files:
        print("No .tr files generated!")
        return False
    
    # Step 3: Extract unicharset
    print("\n=== Step 3: Extracting unicharset ===")
    box_files = sorted(training_dir.glob("stamp_*.box"))
    cmd = ['unicharset_extractor'] + [str(b) for b in box_files]
    run_cmd(cmd)
    
    unicharset = training_dir / "unicharset"
    if not unicharset.exists():
        print("unicharset not created!")
        return False
    print("unicharset created")
    
    # Step 4: Create font_properties
    print("\n=== Step 4: Creating font_properties ===")
    font_props = training_dir / "font_properties"
    with open(font_props, 'w') as f:
        f.write("stamp 0 0 0 0 0\n")
    print("font_properties created")
    
    # Step 5: Run mftraining
    print("\n=== Step 5: Running mftraining ===")
    cmd = [
        'mftraining',
        '-F', str(font_props),
        '-U', str(unicharset),
        '-O', f'{model_name}.unicharset'
    ] + [str(tr) for tr in tr_files]
    run_cmd(cmd)
    
    # Step 6: Run cntraining
    print("\n=== Step 6: Running cntraining ===")
    cmd = ['cntraining'] + [str(tr) for tr in tr_files]
    run_cmd(cmd)
    
    # Step 7: Combine into traineddata
    print("\n=== Step 7: Combining traineddata ===")
    cmd = ['combine_tessdata', f'{model_name}.']
    run_cmd(cmd)
    
    # Check if traineddata was created
    traineddata = training_dir / f"{model_name}.traineddata"
    if traineddata.exists():
        print(f"\n✅ Training complete!")
        print(f"Model: {traineddata}")
        
        # Copy to tessdata
        import shutil
        dest = Path(TESSDATA_DIR) / f"{model_name}.traineddata"
        shutil.copy(traineddata, dest)
        print(f"Deployed to: {dest}")
        
        return True
    else:
        print("\n❌ Training failed - no traineddata created")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
