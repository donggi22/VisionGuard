Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[설치] 가상환경 생성 중..."
    python -m venv .venv
}

& .venv\Scripts\Activate.ps1

if (-not (Test-Path ".venv\Lib\site-packages\cv2")) {
    Write-Host "[설치] 패키지 설치 중..."
    pip install -r requirements.txt
}

if (-not (Test-Path ".env")) {
    Write-Error "[오류] .env 파일이 없습니다. .env.example 을 복사해서 .env 를 만들고 DISCORD_WEBHOOK_URL 을 입력하세요."
    exit 1
}

python main.py
