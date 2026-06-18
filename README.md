아키텍처:
```
/home/dev/cctv/
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
├── recordings/
└── captures/
```

실행 방법:
```
cd /home/dev/cctv

# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# Discord Webhook URL 설정
cp .env.example .env
# .env 에서 DISCORD_WEBHOOK_URL 입력

# 환경변수 로드 후 실행
export $(cat .env | grep -v ^# | xargs)

# 매번 환경변수 치기 귀찮으면 run.sh 만들어두면 됩니다:
cat > /home/dev/cctv/run.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export $(cat .env | grep -v ^# | xargs)
python main.py
EOF
chmod +x /home/dev/cctv/run.sh

# 가상환경 활성화 및 서비스 실행
cd /home/dev/cctv
source .venv/bin/activate
python main.py
```

cam 안 열릴 때:

video 그룹 권한 때문입니다.

리눅스에서 /dev/video0 같은 하드웨어 장치 파일은 보안상 아무나 못 씁니다. video 그룹에 속한 유저만 접근 가능한데, 처음 계정 만들 때 video 그룹에 자동으로 안 넣어주는 배포판이 많습니다.

```
crw-rw---- 1 root video /dev/video0
           ↑        ↑
         root만     video 그룹만 읽기/쓰기 가능
```
`sudo usermod -aG video $USER` 로 현재 유저를 `video` 그룹에 추가해줬고, 재로그인(터미널 껐다 켜거나 SSH 재접속)으로 그룹 변경이 세션에 적용된 겁니다.

카메라 테스트:
```
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
print('열림:', cap.isOpened())
ret, frame = cap.read()
print('프레임:', ret, frame.shape if ret else 'FAIL')
cap.release()
"
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