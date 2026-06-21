## 아키텍처

```
VisionGuard/
├── main.py              # 메인 루프 (모든 컴포넌트 통합)
├── config.py            # 환경변수 기반 설정
├── motion_detector.py   # MOG2 배경 차분 모션 감지
├── video_buffer.py      # deque 순환 버퍼 (앞 10초)
├── yolo_detector.py     # YOLOv8n + 쿨타임 기반 새 객체 추적
├── recorder.py          # 이벤트 전후 10초 비동기 녹화 + 자동 정리
├── discord_notifier.py  # Webhook + 이미지 첨부 비동기 전송
├── web_app.py           # FastAPI MJPEG 스트리밍 + 이벤트 대시보드 (로그인 포함)
├── monitor.py           # 상태 감시 (미사용)
├── requirements.txt
├── .env.example
├── run.bat              # Windows 실행 스크립트
├── run.ps1              # Windows PowerShell 실행 스크립트
├── run.sh               # Linux 실행 스크립트
├── scheduled-task.txt   # 작업 스케줄러 명령어 모음
├── recordings/
└── captures/
```

---

## 아키텍처 흐름

```
매 프레임 → 순환버퍼 push
          → MOG2 모션 감지
               ↓ 모션 있을 때만
          → YOLOv8n 추론
               ↓ 새 객체 + 쿨타임 통과 시
          → 캡쳐 저장 + 영상 녹화 시작
          → Discord 알림 (비동기)
          → 웹 대시보드 이벤트 추가
```

---

## Windows 초기 설치

### 1. Python 설치

```powershell
winget install Python.Python.3.12 --source winget
```

winget 오류 시 PowerShell로 직접 다운로드:

```powershell
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile "$env:TEMP\python-installer.exe" -UseBasicParsing
Start-Process "$env:TEMP\python-installer.exe" -ArgumentList "/quiet", "InstallAllUsers=0", "PrependPath=1" -Wait
```

### 2. Visual C++ Redistributable 설치

PyTorch 실행에 필요합니다.

```powershell
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "$env:TEMP\vc_redist.x64.exe" -UseBasicParsing
Start-Process "$env:TEMP\vc_redist.x64.exe" -ArgumentList "/quiet", "/norestart" -Wait
```

### 3. .env 설정

```powershell
copy .env.example .env
```

`.env` 파일을 열어 아래 항목 입력:

| 항목 | 설명 |
|---|---|
| `DISCORD_WEBHOOK_URL` | Discord 채널 설정 → 연동 → 웹후크에서 생성 |
| `LOGIN_USERNAME` | 웹 대시보드 로그인 아이디 |
| `LOGIN_PASSWORD` | 웹 대시보드 로그인 비밀번호 |
| `SESSION_SECRET` | 세션 서명용 랜덤 문자열 (아래 명령으로 생성) |

SESSION_SECRET 생성:
```powershell
-join ((1..32) | % { [char](Get-Random -Min 33 -Max 126) })
```

### 4. 실행

**run.bat 더블클릭** — 가상환경 생성 + 패키지 설치 + 실행 자동 처리

또는 PowerShell:
```powershell
.\run.ps1
```

PowerShell 실행 정책 오류 시:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

수동 실행:
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### 5. 방화벽 설정 (외부 접속 허용)

```powershell
New-NetFirewallRule -DisplayName "VisionGuard 8080" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow -Profile Any
```

---

## 부팅 시 자동 시작 등록

`프로그램 관리 명령어.txt` 파일에 전체 명령어 모음이 있습니다.

```powershell
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\dev\Desktop\VisionGuard\.venv\Scripts\python.exe" `
    -Argument "main.py" `
    -WorkingDirectory "C:\Users\dev\Desktop\VisionGuard"

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)

$principal = New-ScheduledTaskPrincipal `
    -UserId "dev" `
    -LogonType S4U `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "VisionGuard" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force
```

---

## 웹 대시보드

- 주소: `http://내부IP:8080` (예: `http://192.168.0.9:8080`)
- 로그인 후 실시간 영상 스트리밍 및 감지 이벤트 확인
- 우측 상단 로그아웃 버튼

---

## Discord 알림

- 가동 시: `✅ VisionGuard 가동됨`
- 감지 시: 캡쳐 이미지 + 감지된 클래스명

---

## 카메라 테스트

```powershell
.venv\Scripts\python -c "
import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('열림:', cap.isOpened())
ret, frame = cap.read()
print('프레임:', ret, frame.shape if ret else 'FAIL')
cap.release()
"
```
