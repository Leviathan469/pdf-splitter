import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")
TESSERACT_EXE = os.path.join(TESSERACT_DIR, "tesseract.exe")
OUTPUT_DIR = Path("lstm_training")
MODEL_NAME = "stamp"

def generate_lstmf(training_output_dir, output_dir):
    """Generate .lstmf files from stamp images using tesseract's lstm.train mode."""
    training_output_dir = Path(training_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add tesseract to PATH
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    
    images = sorted(training_output_dir.glob("stamp_*.tif"))
    print(f"Found {len(images)} TIFF files")
    
    for tiff in images:
        lstmf = output_dir / f"{tiff.stem}.lstmf"
        cmd = ['tesseract', str(tiff), str(lstmf.with_suffix('')), 'lstm.train']
        
        print(f"  Generating {lstmf.name}...")
        result = os.system(' '.join(cmd))
        if result == 0 and lstmf.exists():
            print(f"    OK")
        else:
            print(f"    FAILED")
    
    lstmf_files = list(output_dir.glob("*.lstmf"))
    print(f"\nGenerated {len(lstmf_files)} .lstmf files")
    return lstmf_files

def main():
    generate_lstmf("training_output", OUTPUT_DIR)

if __name__ == "__main__":
    main()
