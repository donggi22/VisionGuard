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
    VERIFY_WINDOW_SECONDS, VERIFY_SAMPLE_COUNT,
    CAMERA_RETRY_SECONDS, CAMERA_REOPEN_AFTER_FAILS, RESUME_DETECT_SECONDS,
    STOP_FLAG_PATH,
)
from discord_notifier import DiscordNotifier
from motion_detector import MotionDetector
from power_events import start_power_watcher
from recorder import EventRecorder
from video_buffer import BufferedFrame, VideoBuffer
from yolo_detector import YoloDetector

# 트리거 시각 기준 샘플링 오프셋(초). 예: 6개 -> [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
_VERIFY_OFFSETS = np.linspace(
    -VERIFY_WINDOW_SECONDS, VERIFY_WINDOW_SECONDS, VERIFY_SAMPLE_COUNT
).tolist()
_PRE_OFFSETS = [o for o in _VERIFY_OFFSETS if o <= 0]
_POST_OFFSETS = [o for o in _VERIFY_OFFSETS if o > 0]


def _select_pre_samples(
    pre_frames: list[BufferedFrame], trigger_time: float, offsets: list[float]
) -> list[np.ndarray]:
    """과거 프레임(video_buf 스냅샷) 중 각 목표 시각에 가장 가까운 프레임을 뽑는다."""
    if not pre_frames:
        return []
    samples = []
    for off in offsets:
        target = trigger_time + off
        nearest = min(pre_frames, key=lambda bf: abs(bf.timestamp - target))
        samples.append(nearest.frame)
    return samples


# 영구 보관으로 변경하여 비활성화
# def _cleanup_loop(recorder: EventRecorder):
#     """매일 자정에 오래된 파일 정리."""
#     while True:
#         time.sleep(3600)
#         recorder.cleanup_old_files()


def _open_camera(backend: int) -> cv2.VideoCapture | None:
    """
    카메라가 열릴 때까지 재시도한다.
    빠른 시작 부팅 직후에는 Wi-Fi 연결이 작업 실행보다 늦어 RTSP 연결이 실패하므로,
    종료하지 않고 네트워크/카메라가 준비될 때까지 대기한다.
    대기 중 종료 요청(STOP_FLAG_PATH)이 들어오면 None을 반환한다.
    """
    attempt = 0
    while True:
        if STOP_FLAG_PATH.exists():
            return None
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        if cap.isOpened():
            if attempt:
                print(f"[CCTV] 카메라 연결됨 (재시도 {attempt}회)")
            return cap
        cap.release()
        attempt += 1
        print(f"[경고] 카메라 열기 실패, {CAMERA_RETRY_SECONDS}초 후 재시도 ({attempt}회)")
        time.sleep(CAMERA_RETRY_SECONDS)


def _start_web_server():
    uvicorn.run(
        web_app.app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level="warning",
    )


def _verify_and_notify(
    yolo: YoloDetector,
    frames: list[np.ndarray],
    capture_path: Path,
    discord: DiscordNotifier,
):
    """
    이미 캡처/녹화가 끝난 트리거 시점 전후 프레임들을 뒤늦게 YOLO로 검증한다.
    CPU 추론 지연이 캡처 타이밍에 영향을 주지 않도록 별도 스레드에서 실행되며,
    여러 프레임 중 하나라도 ALERT_CLASSES 객체가 확인되면(early-exit) 알림/로그를 남긴다.
    이렇게 하는 이유: 트리거 순간 프레임 1장만 보면 객체가 가려지거나 화면 끝에
    걸쳐 있어 놓칠 수 있어서, 트리거 전후 ±VERIFY_WINDOW_SECONDS 구간을 함께 확인한다.
    """
    detections: list = []
    matched_frame = None
    for f in frames:
        try:
            detections = yolo.detect(f)
        except Exception as e:
            # YOLO 추론 자체가 실패해도(모델 설정 오류 등) 검증 전체가 조용히 죽지 않도록
            # 로그를 남기고 나머지 프레임은 계속 시도한다.
            print(f"  └ YOLO 추론 오류, 다음 프레임으로 계속: {e}")
            detections = []
            continue
        if detections:
            matched_frame = f
            break

    if not detections:
        print(f"  └ YOLO 미확인 ({len(frames)}프레임 검증) — 오탐으로 판단, 알림 생략")
        return

    labels = [d.label for d in detections]
    label_str = "+".join(labels)
    annotated = yolo.draw(matched_frame.copy(), detections, {d.label for d in detections})

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
    print(f"[CCTV] 시작 | 웹 http://{WEB_HOST}:{WEB_PORT}")

    # 컴포넌트 초기화
    motion_det = MotionDetector()
    video_buf = VideoBuffer()
    yolo = YoloDetector()
    recorder = EventRecorder()
    discord = DiscordNotifier()

    # 이전 실행에서 남은 종료 요청 파일이 있으면 시작하자마자 종료되므로 지운다
    STOP_FLAG_PATH.unlink(missing_ok=True)
    cap = None

    def _stop():
        """정상 종료: 알림 전송 → 진행 중인 녹화 마무리 → 카메라 해제."""
        print("\n[CCTV] 종료 중...")
        STOP_FLAG_PATH.unlink(missing_ok=True)
        discord.notify_status("⏹️ VisionGuard 종료됨")
        recorder.stop()
        if cap is not None:
            cap.release()
        sys.exit(0)

    # 카메라는 무거운 초기화(YOLO 로딩 등)가 끝난 뒤에 연다.
    # 먼저 열면 초기화 동안 스트림 프레임이 버퍼에 쌓여 영상이 그만큼 지연된다.
    # DSHOW는 장치 번호(USB 카메라)만 열 수 있으므로 RTSP 등 URL은 FFMPEG로 연다
    is_stream = isinstance(CAMERA_INDEX, str)
    backend = cv2.CAP_FFMPEG if is_stream else cv2.CAP_DSHOW
    cap = _open_camera(backend)
    if cap is None:
        _stop()

    discord.notify_status("▶️ VisionGuard 가동됨")

    # 웹 서버 백그라운드 스레드
    web_thread = threading.Thread(target=_start_web_server, daemon=True)
    web_thread.start()

    # 정리 스레드 (영구 보관으로 변경하여 비활성화)
    # cleanup_thread = threading.Thread(target=_cleanup_loop, args=(recorder,), daemon=True)
    # cleanup_thread.start()

    # Ctrl+C 처리 (직접 실행한 콘솔 창에서 종료할 때)
    def _sigint(sig, frame):
        _stop()

    signal.signal(signal.SIGINT, _sigint)

    # 절전/빠른 시작 종료 알림. Windows가 오래 기다려주지 않으므로 짧은 타임아웃으로 보낸다
    start_power_watcher(
        on_suspend=lambda: discord.notify_status("⏹️ VisionGuard 종료됨 (절전/빠른 시작)", timeout=2),
    )

    frame_interval = 1.0 / FPS
    _last_motion_alert = [0.0]  # list로 감싸서 중첩 스코프에서 수정 가능하게
    pending_verify: dict | None = None  # 트리거 후 post 샘플을 모으는 중이면 값이 채워짐

    print("[CCTV] 루프 시작. Ctrl+C 로 종료.")

    read_fails = 0
    # 빠른 시작/절전 시 프로세스는 종료되지 않고 멈췄다가 그대로 이어서 실행되므로
    # 작업 스케줄러가 재실행하지 않는다. time.monotonic()은 절전·최대 절전 시간을
    # 세지 않으므로 벽시계와의 차이로 멈춰 있던 시간을 계산해 복귀를 감지한다.
    wall_prev, mono_prev = time.time(), time.monotonic()

    while True:
        loop_start = time.time()

        # 중지.bat 등의 종료 요청 (작업 스케줄러 실행 중에는 Ctrl+C를 보낼 수 없으므로 파일로 요청받는다)
        if STOP_FLAG_PATH.exists():
            _stop()

        mono_now = time.monotonic()
        suspended = (loop_start - wall_prev) - (mono_now - mono_prev)
        wall_prev, mono_prev = loop_start, mono_now
        if suspended >= RESUME_DETECT_SECONDS:
            print(f"[CCTV] 절전/빠른 시작 복귀 감지 ({suspended:.0f}초 정지), 카메라 재연결")
            # 멈춰 있던 동안 RTSP 연결이 끊겼거나 오래된 프레임이 쌓여 있으므로 다시 연다.
            # 카메라가 열렸다면 네트워크도 연결된 것이므로 그 뒤에 알림을 보낸다.
            cap.release()
            cap = _open_camera(backend)
            if cap is None:
                _stop()
            read_fails = 0
            pending_verify = None
            discord.notify_status("🔄 VisionGuard 복귀됨")
            wall_prev, mono_prev = time.time(), time.monotonic()
            continue

        ret, raw = cap.read()
        if not ret:
            read_fails += 1
            print("[경고] 프레임 읽기 실패, 재시도...")
            # RTSP는 끊긴 뒤 같은 캡처 객체로는 복구되지 않으므로 연속 실패 시 다시 연다
            if read_fails >= CAMERA_REOPEN_AFTER_FAILS:
                print("[경고] 연속 읽기 실패, 카메라 재연결 시도")
                cap.release()
                cap = _open_camera(backend)
                if cap is None:
                    _stop()
                read_fails = 0
                continue
            time.sleep(0.5)
            continue
        read_fails = 0

        # 저장(캡처·녹화)과 YOLO 검증은 카메라 원본 해상도(raw)로,
        # 모션 감지와 웹 스트리밍은 축소 해상도(frame)로 처리해 CPU 부하를 유지한다
        frame = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))

        # 순환 버퍼에 항상 push
        video_buf.push(raw)

        # 녹화 중이면 recorder의 프레임 큐에도 push
        recorder.push_frame(raw)

        # 진행 중인 YOLO 검증이 있으면, 목표 시각이 지난 post 프레임을 채집
        if pending_verify is not None:
            targets = pending_verify["post_targets"]
            now = time.time()
            while targets and now >= targets[0]:
                pending_verify["post_samples"].append(raw.copy())
                targets.pop(0)
            if not targets:
                all_pairs = pending_verify["pre_pairs"] + list(
                    zip(_POST_OFFSETS, pending_verify["post_samples"])
                )
                all_pairs.sort(key=lambda p: abs(p[0]))  # 트리거에 가까운 프레임부터 검증
                threading.Thread(
                    target=_verify_and_notify,
                    args=(yolo, [f for _, f in all_pairs], pending_verify["capture_path"], discord),
                    daemon=True,
                ).start()
                pending_verify = None

        # --- 모션 감지 ---
        motion, mask = motion_det.update(frame)
        display = frame.copy()
        motion_det.draw_roi(display)

        now = time.time()
        if motion and (now - _last_motion_alert[0]) >= ALERT_COOLDOWN_SECONDS:
            trigger_time = now
            _last_motion_alert[0] = trigger_time
            trigger_frame = raw.copy()

            # 1) 선(先) 캡처: YOLO 결과를 기다리지 않고 즉시 캡쳐 + 녹화 시작/연장
            #    (CPU 추론 지연 때문에 객체가 지나간 뒤에 캡쳐되는 것을 방지)
            capture_path = recorder.save_capture(trigger_frame, "motion")
            pre_frames = video_buf.snapshot()
            rec_path = recorder.trigger_recording(pre_frames, "motion")

            status = "신규 녹화" if rec_path is not None else f"기존 녹화 +{POST_RECORD_SECONDS}초 연장"
            print(
                f"[모션] {datetime.now().strftime('%H:%M:%S')} 캡쳐됨 ({status})"
                f" — YOLO 검증 대기 (±{VERIFY_WINDOW_SECONDS:.1f}초, {VERIFY_SAMPLE_COUNT}프레임)"
            )

            # 2) 후(後) YOLO 비동기 검증 준비: 트리거 전후 프레임을 모아 확인된 경우에만 알림/로그 전송
            pre_samples = _select_pre_samples(pre_frames, trigger_time, _PRE_OFFSETS) or [trigger_frame]
            pending_verify = {
                "capture_path": capture_path,
                "pre_pairs": list(zip(_PRE_OFFSETS, pre_samples)),
                "post_targets": [trigger_time + o for o in _POST_OFFSETS],
                "post_samples": [],
            }
            if not pending_verify["post_targets"]:
                # post 오프셋이 없으면(검증 윈도우가 0인 경우) 바로 검증
                all_pairs = sorted(pending_verify["pre_pairs"], key=lambda p: abs(p[0]))
                threading.Thread(
                    target=_verify_and_notify,
                    args=(yolo, [f for _, f in all_pairs], capture_path, discord),
                    daemon=True,
                ).start()
                pending_verify = None

        # 상태 오버레이 표시
        ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, ts_text, (8, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if motion:
            cv2.putText(display, "MOTION", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

        # 웹 스트리밍용 프레임 갱신
        web_app.update_frame(display)

        # FPS 제한 (스트림은 cap.read()가 카메라 송출 속도에 맞춰 대기하므로 제한하지 않는다.
        # 제한하면 일시적 지연으로 쌓인 프레임을 따라잡지 못해 지연이 계속 유지된다)
        if is_stream:
            continue
        elapsed = time.time() - loop_start
        sleep_time = frame_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
