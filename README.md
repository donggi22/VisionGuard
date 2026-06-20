아키텍처:
```
VisionGuard/
├── main.py              # 메인 루프 (모든 컴포넌트 통합)
├── config.py            # 환경변수 기반 설정
├── motion_detector.py   # MOG2 배경 차분 모션 감지
├── video_buffer.py      # deque 순환 버퍼 (앞 10초)
├── yolo_detector.py     # YOLOv8n + 쿨타임 기반 새 객체 추적
├── recorder.py          # 이벤트 전후 10초 비동기 녹화 + 자동 정리
├── discord_notifier.py  # Webhook + 이미지 첨부 비동기 전송
├── web_app.py           # FastAPI MJPEG 스트리밍 + 이벤트 대시보드
├── requirements.txt
├── .env.example
├── run.bat              # Windows 실행 스크립트
├── run.ps1              # Windows PowerShell 실행 스크립트
├── run.sh               # Linux 실행 스크립트
├── recordings/
└── captures/
```

---

## Windows 실행 방법

### 1. Python 설치

```powershell
winget install Python.Python.3.12
```

설치 후 터미널 새로 열고 확인:
```powershell
python --version
```

### 2. .env 설정

```powershell
copy .env.example .env
```

`.env` 파일을 열어 `DISCORD_WEBHOOK_URL` 입력.

### 3. 실행

**방법 A — 배치 파일 더블클릭 (가장 간단)**

`run.bat` 더블클릭. 가상환경 생성 + 패키지 설치 + 실행을 자동으로 합니다.

**방법 B — PowerShell**

```powershell
.\run.ps1
```

> PowerShell 실행 정책 오류 시:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

**방법 C — 수동**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## Linux 실행 방법

```bash
cd /home/dev/cctv

# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# Discord Webhook URL 설정
cp .env.example .env
# .env 에서 DISCORD_WEBHOOK_URL 입력

# 실행
python main.py
```

`run.sh` 사용 시:
```bash
chmod +x run.sh
./run.sh
```

cam 안 열릴 때 (Linux):

video 그룹 권한 때문입니다.

리눅스에서 /dev/video0 같은 하드웨어 장치 파일은 보안상 아무나 못 씁니다. video 그룹에 속한 유저만 접근 가능한데, 처음 계정 만들 때 video 그룹에 자동으로 안 넣어주는 배포판이 많습니다.

```
crw-rw---- 1 root video /dev/video0
           ↑        ↑
         root만     video 그룹만 읽기/쓰기 가능
```
`sudo usermod -aG video $USER` 로 현재 유저를 `video` 그룹에 추가해줬고, 재로그인(터미널 껐다 켜거나 SSH 재접속)으로 그룹 변경이 세션에 적용된 겁니다.

---

카메라 테스트:
```python
import cv2
cap = cv2.VideoCapture(0)
print('열림:', cap.isOpened())
ret, frame = cap.read()
print('프레임:', ret, frame.shape if ret else 'FAIL')
cap.release()
```

<br>

웹 대시보드는 http://localhost:8080 에서 바로 열립니다.

아키텍처 흐름:

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

Discord Webhook URL은 Discord 채널 설정 → 연동 → 웹후크 에서 생성할 수 있습니다.
