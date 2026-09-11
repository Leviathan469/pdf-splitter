# PDF Splitter by Stamp

Split large PDFs into smaller files wherever a RECEIVED stamp appears.

## Quick Start

### Windows (Batch File)
```cmd
split_pdf.bat "C:\path\to\your_document.pdf"
```

### Any OS (Python)
```bash
python split_pdf.py "document.pdf" --template stamp_template.png --output-dir output_folder
```

## What It Does

1. **Converts** each PDF page to an image
2. **Scans** for the RECEIVED stamp using template matching
3. **Splits** the PDF at each stamp location
4. **Outputs** separate PDFs: `document_part001.pdf`, `document_part002.pdf`, etc.

## Requirements

```bash
pip install opencv-python numpy pypdf
```

**Plus Poppler** (for PDF rendering):

| OS | Command |
|----|---------|
| Windows | `winget install oschwartz10612.Poppler` |
| macOS | `brew install poppler` |
| Linux | `apt-get install poppler-utils` |

## Usage

### Basic
```bash
python split_pdf.py input.pdf --template stamp_template.png --output-dir split_output
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--template` | required | Path to stamp template image |
| `--output-dir` | required | Directory for split PDFs |
| `--threshold` | 0.5 | Detection sensitivity (0.0-1.0) |
| `--dpi` | 300 | DPI for PDF rendering |

### Examples
```bash
# More sensitive detection (catches faint stamps)
python split_pdf.py doc.pdf --template stamp.png --output-dir out --threshold 0.4

# Stricter detection (fewer false positives)
python split_pdf.py doc.pdf --template stamp.png --output-dir out --threshold 0.7

# Lower DPI for faster processing
python split_pdf.py doc.pdf --template stamp.png --output-dir out --dpi 200
```

## Files

| File | Purpose |
|------|---------|
| `split_pdf.py` | Main script |
| `split_pdf.bat` | Windows batch wrapper |
| `stamp_template.png` | Your stamp template |
| `stamp_template.pdf` | Template source (for editing) |

## Using Your Own Stamp

1. Scan a clear impression of your stamp (300+ DPI)
2. Crop to include just the stamp
3. Save as `stamp_template.png`
4. Use `--template your_stamp.png`

## How Detection Works

Uses **OpenCV template matching**:
- Compares the stamp template against each page
- Tries multiple scales (0.3x to 2.0x) for size variation
- Returns confidence score (0.0 = no match, 1.0 = perfect match)
- Pages with confidence ≥ threshold are marked as "stamp found"

## Performance

| Document Size | Pages | Time |
|--------------|-------|------|
| 45-page scan | 45 | ~30 seconds |
| 100-page scan | 100 | ~1 minute |

*Based on 300 DPI scans, depending on CPU speed*

## Troubleshoot

| Problem | Solution |
|---------|----------|
| Stamp not detected | Lower threshold: `--threshold 0.4` |
| False positives | Raise threshold: `--threshold 0.7` |
| Wrong pages split | Try a better/cleaner stamp template |
| Slow processing | Lower DPI: `--dpi 200` |
| Poppler not found | Install via winget/brew/apt |
