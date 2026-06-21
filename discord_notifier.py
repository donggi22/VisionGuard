import io
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from config import DISCORD_WEBHOOK_URL


class DiscordNotifier:
    def __init__(self):
        self._enabled = bool(DISCORD_WEBHOOK_URL)
        if not self._enabled:
            print("[Discord] DISCORD_WEBHOOK_URL not set — notifications disabled.")

    def notify_status(self, message: str):
        if not self._enabled:
            return
        try:
            requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
        except Exception as e:
            print(f"[Discord] 상태 알림 오류: {e}")

    def notify(self, frame: np.ndarray, labels: list[str], capture_path: Path = None):
        if not self._enabled:
            return
        t = threading.Thread(
            target=self._send,
            args=(frame.copy(), labels, capture_path),
            daemon=True,
        )
        t.start()

    def _send(self, frame: np.ndarray, labels: list[str], capture_path: Path):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label_str = ", ".join(labels)
        content = f"🚨 **감지됨** `{label_str}` — {ts}"

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_bytes = io.BytesIO(buf.tobytes())
        img_bytes.name = "capture.jpg"

        files = {"file": ("capture.jpg", img_bytes, "image/jpeg")}
        data = {"content": content}

        try:
            resp = requests.post(
                DISCORD_WEBHOOK_URL,
                data=data,
                files=files,
                timeout=10,
            )
            if resp.status_code not in (200, 204):
                print(f"[Discord] 전송 실패: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            print(f"[Discord] 전송 오류: {e}")
