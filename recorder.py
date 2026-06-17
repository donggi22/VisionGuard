import cv2
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
    return cv2.VideoWriter_fourcc(*"mp4v")


class EventRecorder:
    """이벤트 발생 시 앞 N초(버퍼) + 뒤 N초 영상을 비동기로 저장."""

    def __init__(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    def save_capture(self, frame: np.ndarray, label: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CAPTURES_DIR / f"{ts}_{label}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    def start_event_recording(
        self,
        pre_frames: list[BufferedFrame],
        frame_queue_ref: list,  # main loop이 이후 프레임을 append하는 shared list
        event_label: str,
    ) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RECORDINGS_DIR / f"{ts}_{event_label}.mp4"
        t = threading.Thread(
            target=self._write_video,
            args=(path, pre_frames, frame_queue_ref),
            daemon=True,
        )
        t.start()
        return path

    def _write_video(
        self,
        path: Path,
        pre_frames: list[BufferedFrame],
        frame_queue_ref: list,
    ):
        writer = cv2.VideoWriter(
            str(path), _fourcc(), FPS, (FRAME_WIDTH, FRAME_HEIGHT)
        )
        # 앞쪽 버퍼 쓰기
        for bf in pre_frames:
            writer.write(bf.frame)

        # 뒤쪽 POST_RECORD_SECONDS 동안 새 프레임 수집
        deadline = time.time() + POST_RECORD_SECONDS
        while time.time() < deadline:
            if frame_queue_ref:
                writer.write(frame_queue_ref.pop(0))
            else:
                time.sleep(1 / FPS)

        writer.release()

    def cleanup_old_files(self):
        """RETENTION_DAYS 이상 된 파일 삭제."""
        cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
        for directory in (RECORDINGS_DIR, CAPTURES_DIR):
            for f in directory.iterdir():
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
