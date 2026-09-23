import io
import threading
from datetime import datetime

import cv2
import numpy as np
import requests

from config import DISCORD_WEBHOOK_URL


class DiscordNotifier:
    def __init__(self):
        self._enabled = bool(DISCORD_WEBHOOK_URL)
        if not self._enabled:
            print("[Discord] DISCORD_WEBHOOK_URL not set — notifications disabled.")

    def notify(self, images: list[tuple[str, np.ndarray]], labels: list[str], when: datetime):
        """images: (파일명, 프레임) 목록. 한 메시지에 최대 10장까지 첨부."""
        if not self._enabled:
            return
        t = threading.Thread(
            target=self._send,
            args=(images[:10], labels, when),
            daemon=True,
        )
        t.start()

    def _send(self, images: list[tuple[str, np.ndarray]], labels: list[str], when: datetime):
        ts = when.strftime("%Y-%m-%d %H:%M:%S")
        label_str = ", ".join(labels)
        content = f"🚨 **감지됨** `{label_str}` — {ts} ({len(images)}장)"

        files = {}
        for i, (name, frame) in enumerate(images):
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            files[f"files[{i}]"] = (name, io.BytesIO(buf.tobytes()), "image/jpeg")
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
