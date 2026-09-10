#!/usr/bin/env python3
"""
train_stamp_lstm.py

Fine-tune Tesseract eng model on stamp data using lstmtraining.
This produces a proper Tesseract v5 traineddata that works with OEM 0.

Requirements:
  - Tesseract v5 with training tools installed
  - Python packages: Pillow, numpy, opencv-python, pytesseract
  - Training data: PNG files + box files + gt.txt files

Usage:
  python train_stamp_lstm.py <training_dir> [model_name]
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import Image

TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR = os.path.join(TESSERACT_DIR, "tessdata")

def find_training_data(training_dir):
    """Find all training images and their box files."""
    training_dir = Path(training_dir)
    data = []
    
    for box_file in sorted(training_dir.glob("*.box")):
        stem = box_file.stem
        png_file = training_dir / f"{stem}.png"
        gt_file = training_dir / f"{stem}.gt.txt"
        tif_file = training_dir / f"{stem}.tif"
        
        if png_file.exists() and gt_file.exists():
            data.append({
                'stem': stem,
                'box': box_file,
                'png': png_file,
                'gt': gt_file,
                'tif': tif_file if tif_file.exists() else None
            })
    
    return data


def generate_lstmf(training_dir, output_dir):
    """Generate .lstmf files from training data using tesseract."""
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    
    # Find all box files
    box_files = sorted(training_dir.glob("*.box"))
    
    if not box_files:
        print("No box files found!")
        return []
    
    print(f"Generating .lstmf files from {len(box_files)} box files...")
    
    for box_file in box_files:
        stem = box_file.stem
        tif_file = training_dir / f"{stem}.tif"
        
        # Create TIFF if needed
        if not tif_file.exists():
            png_file = training_dir / f"{stem}.png"
            if png_file.exists():
                img = Image.open(png_file)
                img.save(tif_file, compression='none')
        
        if not tif_file.exists():
            print(f"  SKIP: {stem} (no image)")
            continue
        
        lstmf_file = output_dir / f"{stem}.lstmf"
        
        # Generate .lstmf using tesseract
        cmd = [
            'tesseract', str(tiff_file), str(lstmf_file.with_suffix('')),
            '--psm', '6', 'lstm.train'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and lstmf_file.exists():
            print(f"  OK: {lstmf_file.name}")
        else:
            print(f"  FAIL: {stem}")
            if result.stderr:
                print(f"    {result.stderr[:100]}")
    
    lstmf_files = sorted(output_dir.glob("*.lstmf"))
    print(f"\nGenerated {len(lstmf_files)} .lstmf files")
    return lstmf_files


def train_lstm_model(lstmf_files, output_dir, model_name="stamp"):
    """Fine-tune the eng model using lstmtraining."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    os.environ['TESSDATA_PREFIX'] = TESSDATA_DIR
    
    # Create list of lstmf files
    lstmf_list = output_dir / "training_list.txt"
    with open(lstmf_list, 'w') as f:
        for lf in lstmf_files:
            f.write(f"{lf}\n")
    
    # Find eng.lstm (the model to fine-tune from)
    eng_lstm = None
    for path in [
        Path(TESSDATA_DIR) / "eng.lstm",
        Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata" / "eng.lstm",
    ]:
        if path.exists():
            eng_lstm = path
            break
    
    if not eng_lstm:
        # Try to extract from eng.traineddata
        print("Extracting eng.lstm from eng.traineddata...")
        cmd = ['combine_tessdata', '-e', str(Path(TESSDATA_DIR) / "eng.traineddata"), 'eng_']
        subprocess.run(cmd, capture_output=True)
        if Path("eng_.lstm").exists():
            eng_lstm = Path("eng_.lstm").resolve()
        else:
            print("ERROR: Could not find or extract eng.lstm")
            print("LSTM training requires Tesseract to be built with LSTM support.")
            return False
    
    print(f"Using base model: {eng_lstm}")
    
    # Run lstmtraining
    print("\nRunning lstmtraining (this may take a while)...")
    
    cmd = [
        'lstmtraining',
        '--model_output', str(output_dir / f"{model_name}"),
        '--continue_from', str(eng_lstm),
        '--traineddata', str(Path(TESSDATA_DIR) / "eng.traineddata"),
        '--train_listfile', str(lstmf_list),
        '--max_iterations', '1000',
        '--debug_interval', '10',
    ]
    
    print(f"  CMD: {' '.join(cmd)}")
    
    # Run in background since it takes a while
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"  Return code: {result.returncode}")
    if result.stdout:
        print(f"  STDOUT:\n{result.stdout[-1000:]}")
    if result.stderr:
        print(f"  STDERR:\n{result.stderr[-1000:]}")
    
    # Check for output files
    model_files = list(output_dir.glob(f"{model_name}.*"))
    if model_files:
        print(f"\nGenerated {len(model_files)} model files")
        return True
    else:
        print("\nNo model files generated!")
        return False


def combine_model(model_name, training_dir, output_dir):
    """Combine trained model with eng components into final traineddata."""
    
    os.environ['PATH'] = f"{TESSERACT_DIR};{os.environ.get('PATH', '')}"
    
    output_dir = Path(output_dir)
    
    # Extract eng components
    print("Extracting eng components...")
    cmd = ['combine_tessdata', '-e', str(Path(TESSDATA_DIR) / "eng.traineddata"), 'eng_']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    
    if result.returncode != 0:
        print(f"  Failed: {result.stderr}")
        return False
    
    # Copy stamp model files
    print("Copying stamp model files...")
    for f in Path(training_dir).glob(f"{model_name}.*"):
        import shutil
        shutil.copy(f, output_dir / f.name)
        print(f"  Copied: {f.name}")
    
    # Combine
    print("Combining into traineddata...")
    cmd = ['combine_tessdata', f'{model_name}.']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
    
    if result.returncode == 0:
        traineddata = output_dir / f"{model_name}.traineddata"
        if traineddata.exists():
            print(f"\n✅ SUCCESS! Model: {traineddata} ({traineddata.stat().st_size} bytes)")
            
            # Deploy
            dest = Path(TESSDATA_DIR) / f"{model_name}.traineddata"
            try:
                import shutil
                shutil.copy(traineddata, dest)
                print(f"Deployed to: {dest}")
            except Exception as e:
                print(f"Could not deploy (need admin): {e}")
                print(f"Manually copy: {traineddata} -> {dest}")
            
            return True
    
    print(f"Failed: {result.stderr}")
    return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python train_stamp_lstm.py <training_dir> [model_name]")
        sys.exit(1)
    
    training_dir = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "stamp"
    
    if not Path(training_dir).exists():
        print(f"Directory not found: {training_dir}")
        sys.exit(1)
    
    # Output directory
    output_dir = Path("lstm_model_output")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Tesseract LSTM Training for Stamp Font")
    print("=" * 60)
    
    # Step 1: Find training data
    print("\nStep 1: Finding training data...")
    data = find_training_data(training_dir)
    print(f"  Found {len(data)} training samples")
    
    if not data:
        print("No training data found!")
        sys.exit(1)
    
    # Step 2: Generate .lstmf files
    print("\nStep 2: Generating .lstmf files...")
    lstmf_files = generate_lstmf(training_dir, output_dir / "lstmf")
    
    if not lstmf_files:
        print("No .lstmf files generated!")
        sys.exit(1)
    
    # Step 3: Train LSTM model
    print("\nStep 3: Training LSTM model...")
    success = train_lstm_model(lstmf_files, output_dir, model_name)
    
    if not success:
        print("LSTM training failed!")
        response = input("Continue with legacy training only? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 4: Combine model
    print("\nStep 4: Combining model...")
    success = combine_model(model_name, output_dir, output_dir / "final")
    
    if success:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nUsage:")
        print(f"  pytesseract.image_to_string(img, config='--psm 6 -l {model_name}')")
    else:
        print("\nTraining failed.")


if __name__ == "__main__":
    main()
