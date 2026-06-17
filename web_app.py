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

# 메인 루프에서 이 값을 갱신
_current_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_event_log: list[dict] = []  # 최근 이벤트 (최대 50개)


def update_frame(frame: np.ndarray):
    global _current_frame
    with _frame_lock:
        _current_frame = frame.copy()


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
    while True:
        with _frame_lock:
            frame = _current_frame
        if frame is None:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
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
