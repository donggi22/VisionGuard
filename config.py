import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
CAPTURES_DIR = BASE_DIR / "captures"

# Camera
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FPS = int(os.getenv("FPS", "15"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

# Motion detection
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "3000"))  # contour area px²
MOTION_MIN_FRAMES = int(os.getenv("MOTION_MIN_FRAMES", "3"))   # 연속 N프레임 이상 감지돼야 유효

# YOLO
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.4"))

# Recording
PRE_RECORD_SECONDS = int(os.getenv("PRE_RECORD_SECONDS", "10"))
POST_RECORD_SECONDS = int(os.getenv("POST_RECORD_SECONDS", "10"))

# Alerts
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))

# Storage cleanup
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "7"))

# Web
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

# Auth
LOGIN_USERNAME = os.getenv("LOGIN_USERNAME", "admin")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "changeme")
SESSION_SECRET = os.getenv("SESSION_SECRET", "please-set-a-secret-in-env")

# YOLO 클래스 중 알림 대상 (비어있으면 전체)
# COCO 클래스: person=0, car=2, cat=15, dog=16 등
ALERT_CLASSES = os.getenv("ALERT_CLASSES", "person,car,cat,dog").split(",")

# True: 모션 감지만으로 알림 (YOLO는 어노테이션 용도)
# False: YOLO가 객체를 인식해야만 알림
MOTION_ONLY_ALERT = os.getenv("MOTION_ONLY_ALERT", "false").lower() == "true"
