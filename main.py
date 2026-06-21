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
from config import (
    CAMERA_INDEX, FPS, FRAME_WIDTH, FRAME_HEIGHT, WEB_HOST, WEB_PORT,
    MOTION_ONLY_ALERT,
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

    # 이벤트 발생 시 뒤쪽 프레임을 채워줄 shared queue
    post_frame_queue: list = []

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
    recording_until = 0.0
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

        # post-record 중이면 queue에도 push
        if time.time() < recording_until:
            post_frame_queue.append(frame.copy())

        # --- 모션 감지 ---
        motion, mask = motion_det.update(frame)
        display = frame.copy()
        motion_det.draw_roi(display)

        if motion:
            # --- YOLO 실행 (어노테이션 + 클래스 식별) ---
            detections, new_objects = yolo.detect(frame)
            highlight = {d.label for d in new_objects}
            yolo.draw(display, detections, highlight)

            # 알림 조건 결정
            if MOTION_ONLY_ALERT:
                # 모션 감지만으로 알림, 쿨타임은 yolo의 last_alert 대신 별도 관리
                should_alert = (time.time() - _last_motion_alert[0]) >= 30
                labels = [d.label for d in detections] or ["motion"]
                trigger_frame = frame.copy()
                if detections:
                    yolo.draw(trigger_frame, detections, highlight)
            else:
                should_alert = bool(new_objects)
                labels = [d.label for d in new_objects]
                trigger_frame = yolo.draw(frame.copy(), new_objects, highlight)

            if should_alert:
                label_str = "+".join(labels)
                print(f"[이벤트] {datetime.now().strftime('%H:%M:%S')} 감지: {label_str}")

                if MOTION_ONLY_ALERT:
                    _last_motion_alert[0] = time.time()

                # 캡쳐 저장
                capture_path = recorder.save_capture(trigger_frame, label_str)

                # 이벤트 영상 녹화 시작
                pre_frames = video_buf.snapshot()
                recorder.start_event_recording(pre_frames, post_frame_queue, label_str)
                recording_until = time.time() + 10

                # Discord 알림
                discord.notify(trigger_frame, labels, capture_path)

                # 웹 이벤트 로그 업데이트
                web_app.add_event(label_str, capture_path)

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
