import threading
import time
from typing import Callable

import av
import numpy as np

from video_buffer import EncodedPacket


class CameraStream:
    """RTSP 스트림을 별도 스레드에서 수신.

    - 모든 H.264 패킷을 on_packet 콜백으로 넘김 (녹화용, 재인코딩 없음)
    - 디코딩한 최신 프레임만 보관 → 분석 루프가 느려도 녹화/스트림이 밀리지 않음
    """

    def __init__(
        self,
        url: str,
        on_packet: Callable[[EncodedPacket, av.video.stream.VideoStream], None],
        on_disconnect: Callable[[], None],
    ):
        self._url = url
        self._on_packet = on_packet
        self._on_disconnect = on_disconnect
        self._cond = threading.Condition()
        self._latest: av.VideoFrame | None = None
        self._seq = 0
        self._read_seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=5)

    def read(self, width: int, height: int, timeout: float = 5.0) -> np.ndarray | None:
        """아직 읽지 않은 최신 프레임을 BGR ndarray로 반환. timeout 동안 없으면 None."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq != self._read_seq, timeout):
                return None
            frame = self._latest
            self._read_seq = self._seq
        # 색변환 + 리사이즈를 swscale 한 번으로 처리 (분석 주기에만 수행)
        return frame.to_ndarray(format="bgr24", width=width, height=height)

    def _run(self) -> None:
        while self._running:
            try:
                container = av.open(
                    self._url,
                    options={"rtsp_transport": "tcp", "fflags": "nobuffer"},
                    timeout=(10.0, 10.0),
                )
            except av.FFmpegError as e:
                print(f"[카메라] 연결 실패: {e} — 3초 후 재시도")
                time.sleep(3)
                continue

            stream = container.streams.video[0]
            stream.codec_context.thread_type = "AUTO"
            print(
                f"[카메라] 연결됨 — {stream.codec_context.name} "
                f"{stream.codec_context.width}x{stream.codec_context.height}"
            )
            try:
                for packet in container.demux(stream):
                    if not self._running:
                        break
                    if packet.size == 0:  # EOF flush 패킷
                        continue
                    self._on_packet(
                        EncodedPacket(
                            data=bytes(packet),
                            is_keyframe=packet.is_keyframe,
                            wall_time=time.time(),
                        ),
                        stream,
                    )
                    for frame in packet.decode():
                        with self._cond:
                            self._latest = frame
                            self._seq += 1
                            self._cond.notify_all()
                if self._running:
                    print("[카메라] 스트림 종료 — 재연결")
            except av.FFmpegError as e:
                print(f"[카메라] 수신 오류: {e} — 재연결")
            finally:
                # 녹화 세션이 이 스트림을 더 이상 참조하지 않도록 먼저 정리
                self._on_disconnect()
                container.close()
            if self._running:
                time.sleep(1)
