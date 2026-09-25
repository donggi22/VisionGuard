# Graceful stop for VisionGuard.
# Creating stop.flag makes main.py send the Discord "stopped" alert, finish any
# in-progress recording and exit by itself. Force-kill only if it does not exit in time.
param([int]$TimeoutSeconds = 20)

$flag = Join-Path $PSScriptRoot 'stop.flag'
$taskName = 'VisionGuard'

function Get-VGProcess {
    # Started directly (start .bat) or via the task: python running main.py
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*main.py*' }
}

function Test-VGRunning {
    $task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    ($task -and $task.State -eq 'Running') -or [bool](Get-VGProcess)
}

if (-not (Test-VGRunning)) {
    Write-Host '[ERROR] VisionGuard is not running.'
    exit 1
}

New-Item -ItemType File -Path $flag -Force | Out-Null
Write-Host 'Stopping VisionGuard (sending Discord alert)...'

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
    if (-not (Test-VGRunning)) {
        Remove-Item $flag -ErrorAction SilentlyContinue
        Write-Host '[OK] VisionGuard stopped.'
        exit 0
    }
}

# No response: fall back to the previous force-kill behavior (no Discord alert in this case)
schtasks /end /tn $taskName 2>&1 | Out-Null
Get-VGProcess | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Remove-Item $flag -ErrorAction SilentlyContinue
Write-Host "[WARN] No response in $TimeoutSeconds s, force-stopped (no Discord alert)."
exit 0
