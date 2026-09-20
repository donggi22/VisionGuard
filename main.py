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
from pathlib import Path

import cv2
import uvicorn

import numpy as np

import web_app
from config import (
    CAMERA_INDEX, FPS, FRAME_WIDTH, FRAME_HEIGHT, WEB_HOST, WEB_PORT,
    ALERT_COOLDOWN_SECONDS, POST_RECORD_SECONDS,
)
from discord_notifier import DiscordNotifier
from motion_detector import MotionDetector
from recorder import EventRecorder
from video_buffer import VideoBuffer
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


def _verify_and_notify(
    yolo: YoloDetector,
    frame: np.ndarray,
    capture_path: Path,
    discord: DiscordNotifier,
):
    """
    이미 캡처/녹화가 끝난 프레임을 뒤늦게 YOLO로 검증한다.
    CPU 추론 지연이 캡처 타이밍에 영향을 주지 않도록 별도 스레드에서 실행되며,
    ALERT_CLASSES에 해당하는 객체가 실제로 확인된 경우에만 알림/로그를 남긴다.
    """
    detections = yolo.detect(frame)
    if not detections:
        print("  └ YOLO 미확인 — 오탐으로 판단, 알림 생략")
        return

    labels = [d.label for d in detections]
    label_str = "+".join(labels)
    annotated = yolo.draw(frame.copy(), detections, {d.label for d in detections})

    # 캡쳐 파일명을 임시 라벨(_motion)에서 확인된 실제 클래스명으로 변경
    primary_label = labels[0]
    renamed_path = capture_path.with_name(
        capture_path.name.replace("_motion.", f"_{primary_label}.")
    )
    try:
        capture_path = capture_path.rename(renamed_path)
    except OSError as e:
        print(f"  └ 캡쳐 파일명 변경 실패 (경로는 원본 유지): {e}")

    print(f"  └ YOLO 확인됨: {label_str} — 알림 전송")
    discord.notify(annotated, labels, capture_path)
    web_app.add_event(label_str, capture_path)


def main():
    print(f"[CCTV] 시작 — 카메라 {CAMERA_INDEX}, 웹 http://{WEB_HOST}:{WEB_PORT}")

    # 컴포넌트 초기화
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        sys.exit(f"[오류] 카메라 {CAMERA_INDEX} 열기 실패")

    motion_det = MotionDetector()
    video_buf = VideoBuffer()
    yolo = YoloDetector()
    recorder = EventRecorder()
    discord = DiscordNotifier()
    discord.notify_status("✅ VisionGuard 가동됨")

    # 웹 서버 백그라운드 스레드
    web_thread = threading.Thread(target=_start_web_server, daemon=True)
    web_thread.start()

    # 정리 스레드
    cleanup_thread = threading.Thread(target=_cleanup_loop, args=(recorder,), daemon=True)
    cleanup_thread.start()

    # Ctrl+C 처리
    def _sigint(sig, frame):
        print("\n[CCTV] 종료 중...")
        discord.notify_status("⏹️ VisionGuard 종료됨")
        cap.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    frame_interval = 1.0 / FPS
    _last_motion_alert = [0.0]  # list로 감싸서 중첩 스코프에서 수정 가능하게

    print("[CCTV] 루프 시작. Ctrl+C 로 종료.")

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[경고] 프레임 읽기 실패, 재시도...")
            time.sleep(0.5)
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # 순환 버퍼에 항상 push
        video_buf.push(frame)

        # 녹화 중이면 recorder의 프레임 큐에도 push
        recorder.push_frame(frame)

        # --- 모션 감지 ---
        motion, mask = motion_det.update(frame)
        display = frame.copy()
        motion_det.draw_roi(display)

        if motion and (time.time() - _last_motion_alert[0]) >= ALERT_COOLDOWN_SECONDS:
            _last_motion_alert[0] = time.time()
            trigger_frame = frame.copy()

            # 1) 선(先) 캡처: YOLO 결과를 기다리지 않고 즉시 캡쳐 + 녹화 시작/연장
            #    (CPU 추론 지연 때문에 객체가 지나간 뒤에 캡쳐되는 것을 방지)
            capture_path = recorder.save_capture(trigger_frame, "motion")
            pre_frames = video_buf.snapshot()
            rec_path = recorder.trigger_recording(pre_frames, "motion")

            status = "신규 녹화" if rec_path is not None else f"기존 녹화 +{POST_RECORD_SECONDS}초 연장"
            print(f"[모션] {datetime.now().strftime('%H:%M:%S')} 캡쳐됨 ({status}) — YOLO 검증 대기")

            # 2) 후(後) YOLO 비동기 검증: 별도 스레드에서 확인된 경우에만 알림/로그 전송
            threading.Thread(
                target=_verify_and_notify,
                args=(yolo, trigger_frame, capture_path, discord),
                daemon=True,
            ).start()

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
