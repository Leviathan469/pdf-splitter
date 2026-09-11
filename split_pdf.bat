@echo off
echo ============================================
echo PDF Splitter - RECEIVED Stamp Detection
echo ============================================
echo.

if "%~1" == "" (
    echo Usage: split_pdf.bat "input.pdf" [threshold]
    echo.
    echo Example:
    echo   split_pdf.bat "C:\Users\aiden\Downloads\scanned_document.pdf"
    echo   split_pdf.bat "C:\Users\aiden\Downloads\scanned_document.pdf" 0.5
    echo.
    pause
    exit /b 1
)

set PDF=%~1
set THRESHOLD=0.4
if not "%~2" == "" set THRESHOLD=%~2

set TEMPLATE=C:\Users\aiden\pdf-splitter\Stamp_NoDate.png
set OUTPUT=%~dpn1_split

echo Input: %PDF%
echo Template: %TEMPLATE%
echo Threshold: %THRESHOLD%
echo Output: %OUTPUT%
echo.

python C:\Users\aiden\pdf-splitter\split_pdf.py "%PDF%" --template "%TEMPLATE%" --threshold %THRESHOLD% --output-dir "%OUTPUT%"

echo.
pause
