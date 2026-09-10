"""Quick test: run stamp detection on the sample image."""
import sys
from pathlib import Path
from PIL import Image

# Add the script directory to path
sys.path.insert(0, str(Path(__file__).parent))
from pdf_split_by_stamp import page_has_stamp

img_path = Path(r"C:\Users\aiden\AppData\Roaming\Hermes\composer-images\composer_2026-09-10_18-37-42-916_197a54.png")

if not img_path.exists():
    print(f"Image not found: {img_path}")
    sys.exit(1)

img = Image.open(img_path)
print(f"Image size: {img.size}")
print(f"Mode: {img.mode}")

result = page_has_stamp(img, debug=True)
print(f"\nResult: {'STAMP DETECTED' if result else 'NO STAMP'}")
