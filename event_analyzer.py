import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

import web_app
from config import (
    FPS, CAPTURE_OFFSETS, ALERT_COOLDOWN_SECONDS, MOTION_ONLY_ALERT,
)
from discord_notifier import DiscordNotifier
from recorder import EventRecorder
from yolo_detector import YoloDetector

# 대상 객체가 없던 분석 직후 재분석까지 쉬는 시간 (흔들리는 나무 등으로 YOLO가 계속 도는 것 방지)
_NO_DETECT_RETRY_SECONDS = 3.0


class EventAnalyzer:
    """모션 감지 시점 기준 앞뒤 프레임 몇 장만 YOLO로 분석해 알림.

    라이브 영상에는 YOLO를 돌리지 않는다. 메인 루프는 push()로 최근 프레임만
    넘기고, 모션이 잡히면 maybe_start()로 분석을 요청한다. 분석은 한 번에 하나만
    백그라운드 스레드에서 수행한다.
    """

    def __init__(self, yolo: YoloDetector, recorder: EventRecorder, discord: DiscordNotifier):
        self._yolo = yolo
        self._recorder = recorder
        self._discord = discord
        self._lock = threading.Lock()
        # 가장 이른 오프셋(-1.0초)을 덮을 만큼 + 여유
        self._history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=int(FPS * (-min(CAPTURE_OFFSETS) + 0.5)) + 1
        )
        self._busy = False
        self._last_alert = 0.0
        self._retry_after = 0.0

    def push(self, frame: np.ndarray, t: float) -> None:
        with self._lock:
            self._history.append((t, frame))

    def maybe_start(self, t: float) -> bool:
        """분석 중이 아니고 알림 쿨타임이 지났으면 t 시점 기준 분석을 시작."""
        with self._lock:
            if (self._busy or t < self._retry_after
                    or t - self._last_alert < ALERT_COOLDOWN_SECONDS):
                return False
            self._busy = True
            past = list(self._history)  # 앞쪽 프레임은 지금 확보
        threading.Thread(target=self._run, args=(t, past), daemon=True).start()
        return True

    def _run(self, t: float, past: list[tuple[float, np.ndarray]]) -> None:
        try:
            self._analyze(t, past)
        except Exception as e:
            print(f"[분석] 오류: {e}")
        finally:
            with self._lock:
                self._busy = False

    def _analyze(self, t: float, past: list[tuple[float, np.ndarray]]) -> None:
        # 가장 늦은 오프셋(+1.0초) 프레임이 들어올 때까지 대기
        wait = t + max(CAPTURE_OFFSETS) + 1 / FPS - time.time()
        if wait > 0:
            time.sleep(wait)
        last_past = past[-1][0] if past else 0.0
        with self._lock:
            pool = past + [h for h in self._history if h[0] > last_past]

        # 각 오프셋에 가장 가까운 프레임 선택 (루프가 느려 같은 프레임이 겹치면 1장만)
        shots: list[tuple[float, np.ndarray]] = []
        used: set[int] = set()
        for off in CAPTURE_OFFSETS:
            i = min(range(len(pool)), key=lambda k: abs(pool[k][0] - (t + off)))
            if i not in used:
                used.add(i)
                shots.append((off, pool[i][1]))

        results = self._yolo.detect([f for _, f in shots])
        labels = sorted({d.label for dets in results for d in dets})

        if not labels:
            # 대상 객체 없음 → 캡쳐 저장·Discord 전송 안 함 (알림 쿨타임도 소모 안 함)
            with self._lock:
                self._retry_after = time.time() + _NO_DETECT_RETRY_SECONDS
            if MOTION_ONLY_ALERT:
                # 모션 전용 모드: 영상 녹화와 웹 로그만 남김
                self._recorder.trigger("motion")
                web_app.add_event("motion", None)
            return
        with self._lock:
            self._last_alert = t

        label_str = "+".join(labels)
        print(f"[이벤트] {datetime.fromtimestamp(t).strftime('%H:%M:%S')} 감지: {label_str}")

        # 이벤트 영상 녹화 시작 (녹화 중이면 연장). 앞 버퍼가 분석 지연을 덮는다.
        self._recorder.trigger(label_str)

        # 가장 확실하게 검출된 1장만 저장·전송 (동률이면 감지 시점에 가까운 것)
        (off, frame), dets = max(
            ((shot, dets) for shot, dets in zip(shots, results) if dets),
            key=lambda x: (max(d.confidence for d in x[1]), -abs(x[0][0])),
        )
        image = self._yolo.draw(frame.copy(), dets)
        stamp = datetime.fromtimestamp(t).strftime("%Y%m%d_%H%M%S")
        path = self._recorder.save_capture(image, label_str, stamp)
        self._discord.notify([(f"{off:+.1f}s.jpg", image)], labels, datetime.fromtimestamp(t))
        web_app.add_event(label_str, path)
