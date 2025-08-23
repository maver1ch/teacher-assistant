@echo off
echo ===================================================
echo       TEACHER ASSISTANT - KHOI DONG UNG DUNG
echo ===================================================
echo.

REM Check if .env exists
if not exist ".env" (
    echo ERROR: File .env khong ton tai!
    echo Vui long tao file .env voi OPENAI_API_KEY truoc khi chay
    echo Xem file .env.example de biet cach tao
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Khoi dong ung dung Teacher Assistant...
echo Ung dung se mo tai: http://localhost:8501
echo.
echo Nhan Ctrl+C de dung ung dung
echo.

streamlit run app.py