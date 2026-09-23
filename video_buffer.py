import time
from collections import deque
from typing import NamedTuple

from config import PRE_RECORD_SECONDS


class EncodedPacket(NamedTuple):
    """카메라가 보낸 H.264 패킷 사본 (재인코딩 없이 그대로 MP4에 remux)."""
    data: bytes
    is_keyframe: bool
    wall_time: float


class PacketBuffer:
    """항상 최근 PRE_RECORD_SECONDS 초 이상 분량의 인코딩된 패킷을 메모리에 유지.

    디코딩 가능한 영상이 되려면 키프레임부터 시작해야 하므로, 버퍼 맨 앞은
    항상 키프레임이고 PRE_RECORD_SECONDS 보다 최대 GOP 1개 분량만큼 더 길 수 있다.
    스레드 안전하지 않음 — 호출자(EventRecorder)가 lock으로 보호한다.
    """

    def __init__(self):
        self._buf: deque[EncodedPacket] = deque()

    def push(self, packet: EncodedPacket) -> None:
        self._buf.append(packet)
        self._trim()

    def _trim(self) -> None:
        # 두 번째 키프레임이 PRE_RECORD_SECONDS 이전이면 첫 GOP는 필요 없음
        cutoff = time.time() - PRE_RECORD_SECONDS
        while True:
            second_key = next(
                (i for i, p in enumerate(self._buf) if i > 0 and p.is_keyframe),
                None,
            )
            if second_key is None or self._buf[second_key].wall_time > cutoff:
                break
            for _ in range(second_key):
                self._buf.popleft()
        # 키프레임을 아직 못 받았다면 앞쪽 패킷은 디코딩 불가
        while self._buf and not self._buf[0].is_keyframe:
            self._buf.popleft()

    def snapshot(self) -> list[EncodedPacket]:
        """현재 버퍼 전체를 리스트로 반환 (이벤트 발생 시 앞 N초 확보용)."""
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
