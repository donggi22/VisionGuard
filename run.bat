@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [setup] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] python not found. Install Python 3.12 first.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

if not exist ".venv\Lib\site-packages\cv2" (
    echo [setup] Installing packages...
    pip install -r requirements.txt
)

if not exist ".env" (
    echo [error] .env file not found. Copy .env.example to .env and set DISCORD_WEBHOOK_URL.
    pause
    exit /b 1
)

python main.py
if errorlevel 1 pause
