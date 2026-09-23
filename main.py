"""
CCTV Smart Surveillance System
실행: python main.py
환경변수: .env 파일 또는 export 로 설정 (config.py 참조)
"""

import signal
import sys
import threading
import time
from datetime import datetime

import cv2
import uvicorn

import web_app
from camera import CameraStream
from config import (
    CAMERA_URL, FPS, FRAME_WIDTH, FRAME_HEIGHT, WEB_HOST, WEB_PORT,
)
from discord_notifier import DiscordNotifier
from event_analyzer import EventAnalyzer
from motion_detector import MotionDetector
from recorder import EventRecorder
from yolo_detector import YoloDetector


def _cleanup_loop(recorder: EventRecorder):
    """매일 자정에 오래된 파일 정리."""
    while True:
        time.sleep(3600)
        recorder.cleanup_old_files()


def _start_web_server():
    uvicorn.run(
        web_app.app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="warning",
    )


def main():
    print(f"[CCTV] 시작 | 웹 http://{WEB_HOST}:{WEB_PORT}")

    if not CAMERA_URL:
        sys.exit("[오류] CAMERA_URL 이 설정되지 않았습니다")

    # 컴포넌트 초기화
    motion_det = MotionDetector()
    yolo = YoloDetector()
    recorder = EventRecorder()
    discord = DiscordNotifier()
    # YOLO는 라이브 영상이 아니라 모션 시점 앞뒤 캡쳐에만 사용
    analyzer = EventAnalyzer(yolo, recorder, discord)

    # 카메라 스레드: 패킷은 recorder로 바로 전달, 디코딩된 최신 프레임만 보관
    cam = CameraStream(CAMERA_URL, recorder.on_packet, recorder.on_disconnect)
    cam.start()

    # 웹 서버 백그라운드 스레드
    web_thread = threading.Thread(target=_start_web_server, daemon=True)
    web_thread.start()

    # 정리 스레드
    cleanup_thread = threading.Thread(target=_cleanup_loop, args=(recorder,), daemon=True)
    cleanup_thread.start()

    # Ctrl+C 처리
    def _sigint(sig, frame):
        print("\n[CCTV] 종료 중...")
        cam.stop()
        recorder.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    frame_interval = 1.0 / FPS

    print("[CCTV] 루프 시작. Ctrl+C 로 종료.")

    while True:
        loop_start = time.time()

        frame = cam.read(FRAME_WIDTH, FRAME_HEIGHT)
        if frame is None:
            print("[경고] 새 프레임 없음, 대기 중...")
            continue
        frame_time = time.time()
        analyzer.push(frame, frame_time)

        # --- 모션 감지 ---
        motion, mask = motion_det.update(frame)
        display = frame.copy()
        motion_det.draw_roi(display)

        if motion and analyzer.maybe_start(frame_time):
            print(f"[모션] {datetime.now().strftime('%H:%M:%S')} 감지 — 앞뒤 프레임 YOLO 분석 시작")

        # 상태 오버레이 표시
        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, ts_text, (8, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if motion:
            cv2.putText(display, "MOTION", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        # 웹 스트리밍용 프레임 갱신
        web_app.update_frame(display)

        # FPS 제한
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
