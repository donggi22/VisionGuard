@echo off
schtasks /end /tn "VisionGuard" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] VisionGuard is not running.
    pause
) else (
    echo [OK] VisionGuard stopped.
    timeout /t 2 >nul
)
