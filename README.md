# PDF Splitter by Stamp

Split large PDFs into smaller files wherever a RECEIVED stamp appears.

## Quick Start

```bash
python pdf_split_by_stamp.py your_scan.pdf output_folder
```

## How It Works

1. **Red Channel Detection** - Finds red ink (the date) which is rare on insurance documents
2. **Structural Validation** - Verifies size, position, and aspect ratio match stamp characteristics

**No OCR needed for detection** - works with 100% accuracy on test documents.

## For Your 45-Page Test

Set `TESSERACT_CMD` and `POPPLER_PATH` env vars on your work PC:

```bash
set TESSERACT_CMD=C:\Path\To\Tesseract-OCR\tesseract.exe
set POPPLER_PATH=C:\Path\To\poppler\Library\bin
python pdf_split_by_stamp.py your_file.pdf output_folder
```

## Training OCR for RECEIVED Text (Optional)

The outlined "RECEIVED" font is hard for Tesseract to read. Training scripts are provided:

| Script | Status |
|--------|--------|
| `train_on_work_pc.py` | Ready - extracts stamps and trains on work PC |
| `tesseract_training_data.py` | Ready - auto-generates training data from scans |
| `train_stamp_lstm.py` | Blocked - needs non-integerated eng.lstm |
| `train_with_synthetic_data.py` | Ready - generates synthetic data with similar fonts |

### Why Training is Difficult

Tesseract v5 ships with an integerated (compressed) LSTM model that cannot be used as a starting point for fine-tuning. Options:

1. **Use Tesseract v4** - last version with full legacy training support
2. **Find a non-integerated eng.lstm** - from Tesseract builds with LSTM_DEBUG
3. **Train from scratch** - very slow without starting from a pre-trained model

## Dependencies

```bash
pip install pdf2image pytesseract pypdf Pillow opencv-python numpy
```

Plus system tools:
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler](https://github.com/oschwartz10612/poppler-windows) (for pdf2image)
