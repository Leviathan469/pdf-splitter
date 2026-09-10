import os
import sys
import ctypes
from pathlib import Path

# Paths
TESSDATA = Path(r"C:\Program Files\Tesseract-OCR\tessdata")
USER_TESSDATA = Path.home() / "AppData" / "Local" / "Tesseract-OCR" / "tessdata"
USER_TESSDATA.mkdir(parents=True, exist_ok=True)

# Extract eng components using combine_tessdata
import subprocess
cmd = [
    str(TESSDATA / "combine_tessdata.exe"),
    "-e",
    str(TESSDATA / "eng.traineddata"),
    "eng_"
]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"Return code: {result.returncode}")
if result.stdout:
    print(f"STDOUT: {result.stdout[:500]}")
if result.stderr:
    print(f"STDERR: {result.stderr[:500]}")

# Copy extracted files to user tessdata
for f in Path(".").glob("eng_*"):
    dest = USER_TESSDATA / f.name
    import shutil
    shutil.copy(f, dest)
    print(f"Copied {f} -> {dest}")
    f.unlink()  # Clean up

# Copy stamp legacy components to user tessdata
stamp_files = Path("training_output").glob("stamp.*")
for f in stamp_files:
    dest = USER_TESSDATA / f.name
    import shutil
    shutil.copy(f, dest)
    print(f"Copied {f} -> {dest}")

print("\nFiles in user tessdata:")
for f in USER_TESSDATA.iterdir():
    print(f"  {f.name} ({f.stat().st_size} bytes)")
