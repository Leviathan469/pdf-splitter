#!/usr/bin/env python3
"""
train_with_tesstrain.py

Train Tesseract using the tesstrain wrapper (handles everything automatically).

Prerequisites:
  pip install tesstrain

Usage:
  python train_with_tesstrain.py
"""

import os
import sys
import subprocess
from pathlib import Path

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")
MODEL_NAME = "stamp"
TRAINING_DIR = Path("training_output_clean")

def main():
    # Add tesseract to PATH
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    
    # Check if tesstrain is installed
    try:
        import tesstrain
        print(f"tesstrain version: {tesstrain.__version__}")
    except ImportError:
        print("tesstrain not installed. Run: pip install tesstrain")
        return False
    
    # Check for training data
    if not TRAINING_DIR.exists():
        print(f"Training directory not found: {TRAINING_DIR}")
        print("Run generate_clean_training.py first")
        return False
    
    # Find training images
    images = list(TRAINING_DIR.glob("*.png"))
    if not images:
        print(f"No PNG files found in {TRAINING_DIR}")
        return False
    
    print(f"Found {len(images)} training images")
    
    # Create tesstrain training command
    cmd = [
        sys.executable, '-m', 'tesstrain',
        '--lang', 'eng',
        '--langdata_dir', str(Path(tesstrain.__file__).parent / 'langdata'),
        '--tessdata_dir', TESSDATA_DIR,
        '--output_dir', str(TRAINING_DIR / 'tesstrain_output'),
        '--overwrite',
        '--max_iterations', '1000',
    ]
    
    # Add training images
    for img in images:
        box = img.with_suffix('.box')
        if box.exists():
            cmd.extend(['--fontlist', f'{img}'])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
