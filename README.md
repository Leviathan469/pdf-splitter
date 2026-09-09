# PDF Splitter
Split a large PDF into smaller PDFs wherever a 'RECEIVED' stamp page appears.

## Setup
1. Install dependencies:
```bash
pip install pdf2image pytesseract pypdf Pillow
```

2. Install Tesseract and Poppler:
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Poppler: https://github.com/oschwartz10612/poppler-windows/releases

3. Configure paths at the top of `pdf_split_by_stamp.py` if not on PATH:
```python
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"
```

## Usage
```bash
python pdf_split_by_stamp.py "input.pdf" [output_dir] [--debug]
```

- `input.pdf` — required
- `output_dir` — optional, defaults to `./split_output`
- `--debug` — show OCR text for each page

Only splits on stamps dated today or within the last 2 days.
