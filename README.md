# PDF Splitter by Stamp

Split large PDFs into smaller files wherever a RECEIVED stamp appears.

## Quick Start

### Option 1: Windows Batch File (Easiest)
```cmd
split_pdf.bat "C:\path\to\your\scanned_document.pdf"
```

### Option 2: Python Script
```bash
python split_pdf.py "input.pdf" --template stamp_template.png --output-dir "output_folder"
```

### Option 3: Custom Threshold
```cmd
split_pdf.bat "input.pdf" 0.6
```

## How It Works

1. **Extracts** each page of the PDF as a PNG image
2. **Scans** each page for the RECEIVED stamp using OpenCV template matching
3. **Identifies** pages where the stamp appears (confidence > threshold)
4. **Splits** the PDF at each stamp location into separate files

**No OCR needed for detection** — uses computer vision template matching for reliable detection.

## Usage Examples

```bash
# Basic usage
python split_pdf.py document.pdf --template stamp_template.png --output-dir split_output

# With custom threshold (lower = more sensitive, higher = stricter)
python split_pdf.py document.pdf --template stamp_template.png --threshold 0.5 --output-dir split_output

# Higher DPI for better detection on high-res scans
python split_pdf.py document.pdf --template stamp_template.png --dpi 300 --output-dir split_output
```

## Files

| File | Purpose |
|------|---------|
| `split_pdf.py` | Main PDF splitting script |
| `split_pdf.bat` | Windows batch file for easy usage |
| `stamp_template.png` | Your stamp template (RECEIVED with date) |

## Dependencies

```bash
pip install opencv-python numpy pypdf
```

Plus:
- [Poppler](https://github.com/oschwartz10612/poppler-windows) (for PDF to PNG conversion)
  - On Windows: `winget install oschwartz10612.Poppler`

## How to Use Your Own Stamp

1. Scan a document with a clear stamp
2. Open the scan in Paint or similar
3. Crop just the stamp region
4. Save as `stamp_template.png`
5. Run the splitter with your custom template

## Training OCR for RECEIVED Text (Optional)

If you also need to READ the stamp text (dates, initials), see the training scripts:

| Script | Purpose |
|--------|---------|
| `train_stamp_with_tesstrain.py` | Docker-based training with tesstrain |

### Why Training is Difficult on Windows

Tesseract v5 produces corrupted `.lstmf` files on Windows. The solution is Docker:

```powershell
cd C:\Users\aiden\tesstrain
docker run --rm -v "${PWD}:/tesstrain" tesstrain bash -c "cd /tesstrain && make training MODEL_NAME=received-stamp TESSDATA=/tesstrain/usr/share/tessdata_best TESSDATA_REPO=_best MAX_ITERATIONS=1000"
```

## Results

Tested on sample documents:
- **Confidence**: 0.97-1.00 (excellent detection)
- **Speed**: ~1 second per page
- **Output**: `document_part001.pdf`, `document_part002.pdf`, etc.

## For Your Work PC

Set environment variables if Tesseract/Poppler are in non-standard paths:

```cmd
set TESSERACT_CMD=C:\Path\To\Tesseract-OCR\tesseract.exe
set POPPLER_PATH=C:\Path\To\poppler\Library\bin
python split_pdf.py your_file.pdf --template stamp_template.png --output-dir output
```
