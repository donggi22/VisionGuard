import cv2
import queue
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from config import (
    FPS, FRAME_WIDTH, FRAME_HEIGHT,
    POST_RECORD_SECONDS, RETENTION_DAYS,
    RECORDINGS_DIR, CAPTURES_DIR,
)
from video_buffer import BufferedFrame


def _fourcc():
    return cv2.VideoWriter_fourcc(*"avc1")


class EventRecorder:
    """이벤트 발생 시 앞 N초(버퍼) + 뒤 N초 영상을 단일 스레드로 안정적으로 저장."""

    def __init__(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=FPS * 30)
        self._record_thread: threading.Thread | None = None
        self._record_deadline: float = 0.0
        self._is_recording: bool = False
        self._lock = threading.Lock()

    def save_capture(self, frame: np.ndarray, label: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CAPTURES_DIR / f"{ts}_{label}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def push_frame(self, frame: np.ndarray):
        """메인 루프에서 매 프레임 호출하여 녹화 큐에 프레임을 전달합니다."""
        if self._is_recording:
            try:
                self._frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    def trigger_recording(
        self,
        pre_frames: list[BufferedFrame],
        event_label: str,
    ) -> Path | None:
        """
        이벤트 발생 시 호출.
        - 이미 녹화 중이면 종료 시간을 연장합니다.
        - 녹화 중이 아니면 새 녹화 스레드를 시작합니다.
        """
        now = time.time()
        with self._lock:
            if self._is_recording:
                self._record_deadline = now + POST_RECORD_SECONDS
                return None

            self._is_recording = True
            self._record_deadline = now + POST_RECORD_SECONDS

            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = RECORDINGS_DIR / f"{ts}_{event_label}.mp4"
            pre_frame_copies = [bf.frame.copy() for bf in pre_frames]

            self._record_thread = threading.Thread(
                target=self._record_worker,
                args=(path, pre_frame_copies),
                daemon=True,
            )
            self._record_thread.start()
            return path

    def _record_worker(self, path: Path, pre_frames: list[np.ndarray]):
        writer = cv2.VideoWriter(
            str(path), _fourcc(), FPS, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        try:
            for frame in pre_frames:
                writer.write(frame)

            while True:
                with self._lock:
                    deadline = self._record_deadline
                if time.time() >= deadline:
                    break
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                    writer.write(frame)
                except queue.Empty:
                    continue
        finally:
            writer.release()
            with self._lock:
                self._is_recording = False

    def cleanup_old_files(self):
        """RETENTION_DAYS 이상 된 파일 삭제."""
        cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
        for directory in (RECORDINGS_DIR, CAPTURES_DIR):
            for f in directory.iterdir():
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
