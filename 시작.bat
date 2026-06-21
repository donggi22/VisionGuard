@echo off
schtasks /run /tn "VisionGuard" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] VisionGuard task not found in Task Scheduler.
    pause
) else (
    echo [OK] VisionGuard started.
    timeout /t 2 >nul
)
