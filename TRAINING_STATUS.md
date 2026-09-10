# Tesseract Training Status

## Current Situation

Tesseract v4.0.0 is installed at `C:\Program Files\Tesseract-OCR` but **cannot run training** because:

| Issue | Details |
|-------|---------|
| Missing DLLs | `unicharset_extractor.exe` crashes (error 0xC0000135) - missing ICU DLLs |
| Incomplete installer | v4.0.0 (March 2019) doesn't include ICU dependencies |
| Background install fails | v4.1.0-elag2019 installer needs admin rights for silent install |

## To Fix (on your work PC)

**Option 1: Run v4.10 installer interactively**
1. Download: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v4.1.0-elag2019.exe
2. Run it (double-click) and install to `C:\Program Files\Tesseract-OCR-v4`
3. This version includes the required ICU DLLs

**Option 2: Copy ICU DLLs**
1. Get ICU 67 DLLs from: https://github.com/unicode-org/icu/releases/tag/release-67-1
2. Copy `icuuc67.dll`, `icuin67.dll`, `icudt67.dll`, `icuio67.dll`, `icule67.dll`, `iculx67.dll`, `icutu67.dll`, `icuid67.dll` to `C:\Program Files\Tesseract-OCR\`

**Option 3: Use v5 with LSTM (not recommended)**
- Would require building Tesseract from source or finding a non-integerated LSTM
- More complex than just using v4

## After Installing v4.10

Run:
```bash
python train_with_tesseract_v4.py "Stamp training data.pdf"
```

This will produce `stamp.traineddata` for use with:
```python
pytesseract.image_to_string(img, config='--psm 6 --oem 1 -l eng+stamp')
```
