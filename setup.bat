@echo off
echo ===================================================
echo       TEACHER ASSISTANT - SETUP SCRIPT
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python khong duoc tim thay!
    echo Vui long cai dat Python 3.9+ truoc: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Kiem tra Python: OK
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [2/4] Tao virtual environment...
    python -m venv venv
) else (
    echo [2/4] Virtual environment da ton tai
)

REM Activate virtual environment
echo [3/4] Kich hoat virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [4/4] Cai dat dependencies...
pip install -r requirements.txt

echo.
echo ===================================================
echo                 SETUP HOAN THANH!
echo ===================================================
echo.
echo BUOC TIEP THEO:
echo 1. Tao file .env voi OPENAI_API_KEY cua ban
echo 2. Chay: run.bat
echo.
pause