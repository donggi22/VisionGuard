@echo off
net session >nul 2>&1
if %errorlevel% neq 0 (
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs -Wait"
    exit /b
)
powershell -Command "$p = Get-WmiObject Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.ExecutablePath -like '*VisionGuard*' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host '[OK] VisionGuard stopped.' } else { Write-Host '[ERROR] VisionGuard is not running.' }"
timeout /t 3 >nul
