@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
if errorlevel 1 (
    pause
) else (
    timeout /t 2 >nul
)
