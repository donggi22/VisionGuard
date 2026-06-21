import asyncio
import io
import secrets
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from config import CAPTURES_DIR, RECORDINGS_DIR, LOGIN_USERNAME, LOGIN_PASSWORD, SESSION_SECRET

CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CCTV Dashboard")


class AuthMiddleware(BaseHTTPMiddleware):
    _public = {"/login"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._public:
            return await call_next(request)
        if not request.session.get("authenticated"):
            return RedirectResponse("/login", status_code=302)
        return await call_next(request)


# SessionMiddleware는 마지막에 추가해야 가장 먼저 실행됨
app.add_middleware(AuthMiddleware)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="strict", https_only=False)

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


@app.get("/login", response_class=HTMLResponse)
def login_page(error: str = ""):
    error_html = '<p class="error">아이디 또는 비밀번호가 틀렸습니다.</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisionGuard 로그인</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📹</text></svg>">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #111; color: #eee; font-family: 'Segoe UI', sans-serif;
         display: flex; align-items: center; justify-content: center; min-height: 100vh; }}
  .card {{ background: #1a1a2e; border-radius: 12px; padding: 40px 36px; width: 100%; max-width: 360px; }}
  .logo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }}
  .logo span {{ font-size: 1.4rem; }}
  .logo h1 {{ font-size: 1.1rem; letter-spacing: 1px; color: #fff; }}
  label {{ display: block; font-size: .78rem; color: #aaa; margin-bottom: 6px; letter-spacing: .5px; }}
  input {{ width: 100%; background: #111; border: 1px solid #333; border-radius: 6px;
           color: #eee; padding: 10px 12px; font-size: .9rem; margin-bottom: 16px; outline: none; }}
  input:focus {{ border-color: #e74c3c; }}
  button {{ width: 100%; background: #e74c3c; color: #fff; border: none; border-radius: 6px;
            padding: 11px; font-size: .95rem; cursor: pointer; margin-top: 4px; }}
  button:hover {{ background: #c0392b; }}
  .error {{ color: #e74c3c; font-size: .8rem; margin-bottom: 14px; text-align: center; }}
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <span>📹</span>
    <h1>VISIONGUARD</h1>
  </div>
  {error_html}
  <form method="post" action="/login">
    <label>아이디</label>
    <input type="text" name="username" autofocus autocomplete="username">
    <label>비밀번호</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit">로그인</button>
  </form>
</div>
</body>
</html>"""


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    ok = secrets.compare_digest(username, LOGIN_USERNAME) and \
         secrets.compare_digest(password, LOGIN_PASSWORD)
    if ok:
        request.session["authenticated"] = True
        return RedirectResponse("/", status_code=302)
    return RedirectResponse("/login?error=1", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _jpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/events")
def events():
    return JSONResponse(_event_log)


@app.get("/api/recordings")
def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    return JSONResponse([
        {
            "name": f.name,
            "url": f"/recordings/{f.name}",
            "size_mb": round(f.stat().st_size / 1024 / 1024, 1),
            "time": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        for f in files
    ])


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CCTV Dashboard</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📹</text></svg>">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: 'Segoe UI', sans-serif; }
  header { background: #1a1a2e; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.2rem; letter-spacing: 1px; flex: 1; }
  .badge { background: #e74c3c; color: #fff; border-radius: 4px; padding: 2px 8px; font-size: .75rem; }
  .logout { color: #aaa; font-size: .8rem; text-decoration: none; padding: 4px 10px;
            border: 1px solid #444; border-radius: 4px; }
  .logout:hover { border-color: #e74c3c; color: #e74c3c; }
  .container { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 16px; max-width: 1400px; margin: auto; }
  .video-wrap { background: #000; border-radius: 8px; overflow: hidden; }
  .video-wrap img { width: 100%; display: block; }
  .panel { background: #1a1a1a; border-radius: 8px; overflow: hidden; max-height: 80vh; display: flex; flex-direction: column; }
  .tabs { display: flex; border-bottom: 1px solid #2a2a2a; }
  .tab { padding: 10px 16px; font-size: .8rem; color: #888; cursor: pointer; letter-spacing: .5px; }
  .tab.active { color: #eee; border-bottom: 2px solid #e74c3c; }
  .tab-content { padding: 12px; overflow-y: auto; flex: 1; display: none; }
  .tab-content.active { display: block; }
  .event { display: flex; gap: 10px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid #2a2a2a; }
  .event img { width: 80px; height: 60px; object-fit: cover; border-radius: 4px; cursor: pointer; }
  .event .meta { font-size: .8rem; flex: 1; }
  .event .label { color: #e74c3c; font-weight: bold; margin-bottom: 2px; }
  .event .time { color: #888; }
  .no-img { width: 80px; height: 60px; background: #2a2a2a; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: .65rem; color: #666; flex-shrink: 0; }
  .rec-item { padding: 10px 0; border-bottom: 1px solid #2a2a2a; cursor: pointer; }
  .rec-item:hover .rec-name { color: #e74c3c; }
  .rec-name { font-size: .82rem; color: #ddd; margin-bottom: 3px; word-break: break-all; }
  .rec-meta { font-size: .74rem; color: #666; display: flex; gap: 12px; }
  .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.85); z-index: 100; align-items: center; justify-content: center; }
  .modal.open { display: flex; }
  .modal-inner { background: #1a1a1a; border-radius: 10px; padding: 16px; max-width: 90vw; width: 800px; }
  .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .modal-title { font-size: .85rem; color: #aaa; word-break: break-all; }
  .modal-close { cursor: pointer; color: #888; font-size: 1.2rem; line-height: 1; padding: 4px 8px; }
  .modal-close:hover { color: #e74c3c; }
  .modal video { width: 100%; border-radius: 6px; background: #000; }
  @media (max-width: 768px) { .container { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <span>📹</span>
  <h1>CCTV 실시간 모니터</h1>
  <span class="badge" id="status">연결 중</span>
  <a href="/logout" class="logout">로그아웃</a>
</header>
<div class="container">
  <div class="video-wrap">
    <img id="feed" src="/video_feed" onload="document.getElementById('status').textContent='LIVE'" onerror="document.getElementById('status').textContent='오프라인'">
  </div>
  <div class="panel">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('events')">감지 이벤트</div>
      <div class="tab" onclick="switchTab('recordings')">녹화 목록</div>
    </div>
    <div id="tab-events" class="tab-content active">
      <div id="events"></div>
    </div>
    <div id="tab-recordings" class="tab-content">
      <div id="recordings"></div>
    </div>
  </div>
</div>

<div class="modal" id="modal" onclick="closeModal(event)">
  <div class="modal-inner">
    <div class="modal-header">
      <span class="modal-title" id="modal-title"></span>
      <span class="modal-close" onclick="closeModal()">✕</span>
    </div>
    <video id="modal-video" controls autoplay></video>
  </div>
</div>

<script>
  function switchTab(name) {
    document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', ['events','recordings'][i] === name));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'recordings') loadRecordings();
  }

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

  async function loadRecordings() {
    try {
      const r = await fetch('/api/recordings');
      const data = await r.json();
      const el = document.getElementById('recordings');
      if (!data.length) { el.innerHTML = '<p style="color:#555;font-size:.8rem;padding:8px 0">녹화 파일 없음</p>'; return; }
      el.innerHTML = data.map(f => `
        <div class="rec-item" onclick="playVideo('${f.url}', '${f.name}')">
          <div class="rec-name">▶ ${f.name}</div>
          <div class="rec-meta">
            <span>${f.time}</span>
            <span>${f.size_mb} MB</span>
          </div>
        </div>`).join('');
    } catch {}
  }

  function playVideo(url, name) {
    document.getElementById('modal-title').textContent = name;
    document.getElementById('modal-video').src = url;
    document.getElementById('modal').classList.add('open');
  }

  function closeModal(e) {
    if (e && e.target !== document.getElementById('modal')) return;
    document.getElementById('modal').classList.remove('open');
    const v = document.getElementById('modal-video');
    v.pause();
    v.src = '';
  }

  loadEvents();
  setInterval(loadEvents, 5000);
</script>
</body>
</html>"""
