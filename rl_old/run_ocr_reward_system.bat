@echo off
echo Starting OCR Reward System for ACT RL Training
echo =============================================

REM Set default parameters
set TESSERACT_PATH=D:\Tesseract-OCR\tesseract.exe
set UNITY_HOST=127.0.0.1
set UNITY_PORT=12345
set OCR_FREQUENCY=10

REM Check if Tesseract exists
if not exist "%TESSERACT_PATH%" (
    echo ERROR: Tesseract not found at %TESSERACT_PATH%
    echo Please install Tesseract OCR or update the path in this script
    pause
    exit /b 1
)

echo Configuration:
echo - Tesseract Path: %TESSERACT_PATH%
echo - Unity Host: %UNITY_HOST%
echo - Unity Port: %UNITY_PORT%
echo - OCR Frequency: %OCR_FREQUENCY% Hz
echo.

REM Navigate to the script directory
cd /d "%~dp0"

REM Run the OCR reward system with GUI
echo Starting OCR Reward System with GUI...
echo Press Ctrl+C to stop
echo.

python ocr_reward_system.py ^
    --tesseract-path "%TESSERACT_PATH%" ^
    --unity-host %UNITY_HOST% ^
    --unity-port %UNITY_PORT% ^
    --ocr-frequency %OCR_FREQUENCY%

pause
