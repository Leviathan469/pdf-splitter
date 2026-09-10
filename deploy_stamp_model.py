#!/usr/bin/env python3
"""
deploy_stamp_model.py

Prepare stamp.traineddata for deployment to work PC.
Creates a self-contained package with all required files.
"""

import os
import sys
import zipfile
from pathlib import Path

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")
USER_TESSDATA = Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata"
TRAINING_DIR = Path("training_output")

def main():
    print("=" * 60)
    print("Stamp Model Deployment Package")
    print("=" * 60)
    
    # Create deployment directory
    deploy_dir = Path("stamp_model_deploy")
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy stamp.traineddata
    src = TRAINING_DIR / "stamp.traineddata"
    if src.exists():
        import shutil
        shutil.copy(src, deploy_dir / "stamp.traineddata")
        print(f"Copied: stamp.traineddata ({src.stat().st_size} bytes)")
    else:
        print("ERROR: stamp.traineddata not found! Training may have failed.")
        return False
    
    # Create installation script
    install_script = deploy_dir / "install_stamp_model.py"
    with open(install_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Install stamp.traineddata for Tesseract.

Run this script after installing Tesseract OCR.

Usage:
  python install_stamp_model.py [--tessdata-dir PATH]
"""

import os
import sys
import shutil
from pathlib import Path

def install_stamp_model():
    # Find tessdata directory
    tessdata_dir = None
    
    # Check common locations
    possible_dirs = [
        Path(r"C:\\Program Files\\Tesseract-OCR\\tessdata"),
        Path(r"C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"),
        Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata",
    ]
    
    # Check environment variable
    if "TESSDATA_PREFIX" in os.environ:
        possible_dirs.insert(0, Path(os.environ["TESSDATA_PREFIX"]))
    
    for d in possible_dirs:
        if d.exists():
            tessdata_dir = d
            break
    
    if tessdata_dir is None:
        print("ERROR: Could not find tessdata directory!")
        print("Please specify with --tessdata-dir")
        return False
    
    print(f"Found tessdata: {tessdata_dir}")
    
    # Copy stamp model
    src = Path(__file__).parent / "stamp.traineddata"
    dst = tessdata_dir / "stamp.traineddata"
    
    if not src.exists():
        print(f"ERROR: {src} not found!")
        return False
    
    shutil.copy(src, dst)
    print(f"Installed: {dst}")
    
    # Verify
    if dst.exists():
        print(f"SUCCESS! Stamp model installed ({dst.stat().st_size} bytes)")
        print("\\nUsage in Python:")
        print('  text = pytesseract.image_to_string(img, config="--psm 6 -l eng+stamp")')
        return True
    else:
        print("Installation failed!")
        return False

if __name__ == "__main__":
    success = install_stamp_model()
    sys.exit(0 if success else 1)
''')
    
    print(f"Created: {install_script}")
    
    # Create README
    readme = deploy_dir / "README.txt"
    with open(readme, 'w') as f:
        f.write("""STAMP OCR MODEL - INSTALLATION INSTRUCTIONS
==========================================

This directory contains a trained Tesseract OCR model for reading
RECEIVED stamps on insurance documents.

Prerequisites:
  - Tesseract OCR installed (v5.x recommended)
  - Python 3.8+ with pytesseract, Pillow, opencv-python

Installation:
  1. Run: python install_stamp_model.py
  2. Or manually copy stamp.traineddata to your tessdata directory:
     - Typically: C:\\Program Files\\Tesseract-OCR\\tessdata\\
     - Or: %LOCALAPPDATA%\\Tesseract-OCR\\tessdata\\

Verification:
  Run this in Python:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    from PIL import Image
    img = Image.open("stamp.png")
    text = pytesseract.image_to_string(img, config="--psm 6 -l eng+stamp")
    print(text)

Usage in pdf_split_by_stamp.py:
  The script will automatically use -l eng+stamp when available.
  No changes needed to the main script!

Troubleshooting:
  - If you get "Error opening data file", make sure TESSDATA_PREFIX is set
  - Try: os.environ['TESSDATA_PREFIX'] = r'C:\\Program Files\\Tesseract-OCR\\tessdata'
  - On work PC without admin, use user tessdata directory
""")
    
    print(f"Created: {readme}")
    
    # Create zip
    zip_path = Path("stamp_model.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in deploy_dir.iterdir():
            zf.write(f, f.name)
            print(f"  Added to zip: {f.name}")
    
    print(f"\nDeployment package: {zip_path} ({zip_path.stat().st_size} bytes)")
    print(f"\nTo deploy to work PC:")
    print(f"  1. Copy {zip_path} to work PC")
    print(f"  2. Extract and run: python install_stamp_model.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
