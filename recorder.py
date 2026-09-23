import cv2
import queue
import threading
import time
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from config import (
    FPS, POST_RECORD_SECONDS, MAX_RECORD_SECONDS, RETENTION_DAYS,
    RECORDINGS_DIR, CAPTURES_DIR,
)
from video_buffer import EncodedPacket, PacketBuffer

_STOP = object()  # 세션 종료 신호
_TIME_BASE = Fraction(1, 90000)
# 등간격 타임스탬프가 실제 수신 시각과 이만큼 벌어지면 수신 시각으로 재정렬
_RESYNC_SECONDS = 0.5


class _Session:
    """녹화 파일 1개. 패킷은 전용 큐로 받고 전용 스레드가 MP4에 remux."""

    def __init__(self, path: Path, template: av.video.stream.VideoStream,
                 pre_packets: list[EncodedPacket], now: float):
        self.path = path
        self.started = now
        self.deadline = now + POST_RECORD_SECONDS
        self.queue: queue.Queue = queue.Queue()
        rate = template.average_rate or template.guessed_rate or FPS
        self._step = int(round(1 / (rate * _TIME_BASE)))
        # 입력 스트림의 codecpar는 재연결 시 해제되므로 세션 생성 시점에 복사
        self._out = av.open(str(path), "w", format="mp4",
                            options={"movflags": "+faststart"})
        self._ostream = self._out.add_stream_from_template(template)
        self._t0: float | None = None
        self._last_ts: int | None = None
        for p in pre_packets:
            self.queue.put(p)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.queue.put(_STOP)

    def _write(self) -> None:
        try:
            while (p := self.queue.get()) is not _STOP:
                self._mux(p)
        except Exception as e:
            print(f"[녹화] {self.path.name} 쓰기 오류: {e}")
        finally:
            try:
                self._out.close()
            except Exception as e:
                print(f"[녹화] {self.path.name} 마무리 오류: {e}")
        print(f"[녹화] 저장 완료: {self.path.name}")

    def _timestamp(self, wall_time: float) -> int:
        # 카메라 RTP 타임스탬프는 중복·역행이 잦아 쓰지 않는다. 패킷 1개 = 프레임 1장
        # (B-frame 없음)이므로 등간격으로 찍되, 프레임 누락·fps 변동으로 실제 시각과
        # 벌어지면 수신 시각에 다시 맞춘다.
        if self._t0 is None:
            self._t0 = wall_time
            return 0
        ts = self._last_ts + self._step
        wall_ts = int((wall_time - self._t0) / _TIME_BASE)
        if abs(ts - wall_ts) > _RESYNC_SECONDS / _TIME_BASE:
            ts = max(wall_ts, self._last_ts + 1)
        return ts

    def _mux(self, p: EncodedPacket) -> None:
        ts = self._timestamp(p.wall_time)
        self._last_ts = ts
        pkt = av.Packet(p.data)
        pkt.pts = pkt.dts = ts
        pkt.time_base = _TIME_BASE
        pkt.is_keyframe = p.is_keyframe
        pkt.stream = self._ostream
        self._out.mux(pkt)


class EventRecorder:
    """이벤트 전 N초(패킷 버퍼) + 후 N초를 재인코딩 없이 MP4로 저장.

    녹화 중 새 이벤트가 오면 파일을 새로 만들지 않고 종료 시각만 연장한다
    (최대 MAX_RECORD_SECONDS). 카메라 스레드의 on_packet과 메인 루프의
    trigger가 같은 lock을 쓰므로 스냅샷과 이후 패킷 사이에 누락·중복이 없다.
    """

    def __init__(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer = PacketBuffer()
        self._stream: av.video.stream.VideoStream | None = None
        self._session: _Session | None = None
        self._writers: list[threading.Thread] = []

    def save_capture(self, frame: np.ndarray, label: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = CAPTURES_DIR / f"{ts}_{label}.jpg"
        cv2.imwrite(str(path), frame)
        return path

    # --- 카메라 스레드에서 호출 ---

    def on_packet(self, packet: EncodedPacket, stream: av.video.stream.VideoStream) -> None:
        with self._lock:
            self._stream = stream
            self._buffer.push(packet)
            s = self._session
            if s is None:
                return
            if packet.wall_time >= s.deadline:
                s.finish()
                self._session = None
            else:
                s.queue.put(packet)

    def on_disconnect(self) -> None:
        """스트림이 끊기면 타임스탬프가 이어지지 않으므로 현재 파일을 마감."""
        with self._lock:
            self._end_session()
            self._buffer.clear()
            self._stream = None

    # --- 메인 루프에서 호출 ---

    def trigger(self, event_label: str) -> Path | None:
        """이벤트 녹화 시작 또는 연장. 저장될 파일 경로를 반환."""
        now = time.time()
        with self._lock:
            s = self._session
            if s is not None:
                if now + POST_RECORD_SECONDS <= s.started + MAX_RECORD_SECONDS:
                    s.deadline = now + POST_RECORD_SECONDS
                    return s.path
                self._end_session()  # 최대 길이 도달 → 새 파일로 분할
            if self._stream is None or not len(self._buffer):
                print("[녹화] 카메라 패킷이 아직 없어 녹화를 건너뜀")
                return None
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = RECORDINGS_DIR / f"{ts}_{event_label}.mp4"
            try:
                self._session = _Session(path, self._stream, self._buffer.snapshot(), now)
                self._writers = [t for t in self._writers if t.is_alive()]
                self._writers.append(self._session.thread)
            except Exception as e:
                print(f"[녹화] {path.name} 생성 실패: {e}")
                return None
            return path

    def stop(self) -> None:
        """종료 시 진행 중인 파일을 정상 마감 (moov atom 기록)."""
        with self._lock:
            self._end_session()
            writers = list(self._writers)
        for t in writers:
            t.join(timeout=10)

    def _end_session(self) -> None:
        if self._session is not None:
            self._session.finish()
            self._session = None

    def cleanup_old_files(self):
        """RETENTION_DAYS 이상 된 파일 삭제."""
        cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
        for directory in (RECORDINGS_DIR, CAPTURES_DIR):
            for f in directory.iterdir():
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
