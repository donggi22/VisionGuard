import asyncio
import io
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import CAPTURES_DIR, RECORDINGS_DIR

app = FastAPI(title="CCTV Dashboard")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")
app.mount("/recordings", StaticFiles(directory=str(RECORDINGS_DIR)), name="recordings")

# 메인 루프가 새 프레임을 넣으면 조건변수로 모든 접속자에게 알림.
# JPEG 인코딩은 새 프레임당 1번만 (첫 접속자가 수행, 나머지는 캐시 공유) →
# 접속자 수와 무관하게 인코딩 CPU 고정, 시청자가 없으면 인코딩 안 함.
_frame_cond = threading.Condition()
_raw_frame: np.ndarray | None = None
_raw_seq = 0
_jpeg: bytes | None = None
_jpeg_seq = 0
_event_log: list[dict] = []  # 최근 이벤트 (최대 50개)


def update_frame(frame: np.ndarray):
    """frame은 호출 후 수정하지 않아야 함 (복사 없이 참조만 보관)."""
    global _raw_frame, _raw_seq
    with _frame_cond:
        _raw_frame = frame
        _raw_seq += 1
        _frame_cond.notify_all()


def _next_jpeg(last_seq: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
    """last_seq 이후의 새 프레임 JPEG를 기다려 반환. timeout 동안 없으면 (None, last_seq)."""
    global _jpeg, _jpeg_seq
    with _frame_cond:
        if not _frame_cond.wait_for(lambda: _raw_seq != last_seq, timeout):
            return None, last_seq
        if _jpeg_seq != _raw_seq:
            _, buf = cv2.imencode(".jpg", _raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            _jpeg, _jpeg_seq = buf.tobytes(), _raw_seq
        return _jpeg, _jpeg_seq


def add_event(label: str, capture_path: Path):
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "capture": f"/captures/{capture_path.name}" if capture_path else None,
    }
    _event_log.insert(0, entry)
    if len(_event_log) > 50:
        _event_log.pop()


def _jpeg_generator():
    seq = 0
    while True:
        jpeg, seq = _next_jpeg(seq)
        if jpeg is None:
            continue  # 카메라 정지 등으로 새 프레임 없음 — 바쁜 대기 없이 다시 대기
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _jpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
def events():
    return JSONResponse(_event_log)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CCTV Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: 'Segoe UI', sans-serif; }
  header { background: #1a1a2e; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; letter-spacing: 1px; }
  .badge { background: #e74c3c; color: #fff; border-radius: 4px; padding: 2px 8px; font-size: .75rem; }
  .container { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: auto; }
  .video-wrap { background: #000; border-radius: 8px; overflow: hidden; }
  .video-wrap img { width: 100%; display: block; }
  .panel { background: #1a1a1a; border-radius: 8px; padding: 12px; overflow-y: auto; max-height: 80vh; }
  .panel h2 { font-size: .9rem; color: #aaa; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .event { display: flex; gap: 10px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid #2a2a2a; }
  .event img { width: 80px; height: 60px; object-fit: cover; border-radius: 4px; cursor: pointer; }
  .event .meta { font-size: .8rem; }
  .event .label { color: #e74c3c; font-weight: bold; margin-bottom: 2px; }
  .event .time { color: #888; }
  .no-img { width: 80px; height: 60px; background: #2a2a2a; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: .65rem; color: #666; }
  @media (max-width: 768px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span>📹</span>
  <h1>CCTV 실시간 모니터</h1>
  <span class="badge" id="status">연결 중</span>
</header>
<div class="container">
  <div class="video-wrap">
    <img id="feed" src="/video_feed" onload="document.getElementById('status').textContent='LIVE'" onerror="document.getElementById('status').textContent='오프라인'">
  </div>
  <div class="panel">
    <h2>감지 이벤트</h2>
    <div id="events"></div>
  </div>
</div>
<script>
  async function loadEvents() {
    try {
      const r = await fetch('/events');
      const data = await r.json();
      const el = document.getElementById('events');
      if (!data.length) { el.innerHTML = '<p style="color:#555;font-size:.8rem;padding:8px 0">아직 이벤트 없음</p>'; return; }
      el.innerHTML = data.map(e => `
        <div class="event">
          ${e.capture ? `<img src="${e.capture}" onclick="window.open(this.src)" title="클릭하여 원본 보기">` : '<div class="no-img">캡쳐없음</div>'}
          <div class="meta">
            <div class="label">${e.label}</div>
            <div class="time">${e.time}</div>
          </div>
        </div>`).join('');
    } catch {}
  }
  loadEvents();
  setInterval(loadEvents, 5000);
</script>
</body>
</html>"""
