import time
from collections import deque
from typing import NamedTuple
import numpy as np

from config import FPS, PRE_RECORD_SECONDS


class BufferedFrame(NamedTuple):
    frame: np.ndarray
    timestamp: float


class VideoBuffer:
    """항상 최근 PRE_RECORD_SECONDS 초 분량의 프레임을 메모리에 유지."""

    def __init__(self):
        maxlen = FPS * PRE_RECORD_SECONDS
        self._buf: deque[BufferedFrame] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray) -> None:
        self._buf.append(BufferedFrame(frame=frame.copy(), timestamp=time.time()))

    def snapshot(self) -> list[BufferedFrame]:
        """현재 버퍼 전체를 리스트로 반환 (이벤트 발생 시 앞 10초 확보용)."""
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
