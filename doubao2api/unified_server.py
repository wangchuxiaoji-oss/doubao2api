"""
Unified API server for Doubao (Playwright browser-based).

Exposes OpenAI-compatible endpoints:
  POST /v1/chat/completions     (chat, streaming & non-streaming)
  GET  /v1/models               (list available models)
  GET  /health                  (health check)
  GET  /auth                    (QR login page)

Start with:
    python -m doubao2api
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .browser_client import BrowserClient

log = logging.getLogger("doubao_unified")

# ── Model definitions ────────────────────────────────────────

CHAT_MODELS: Dict[str, bool] = {
    "doubao": False,
    "doubao-pro": False,
    "doubao-think": True,
}

ALL_MODELS = [
    {"id": m, "object": "model", "owned_by": "doubao", "created": 0}
    for m in CHAT_MODELS
]

# ── Request log ring buffer ───────────────────────────────────

_REQUEST_LOG: collections.deque = collections.deque(maxlen=100)
_SERVER_START_TIME: float = time.time()


# ── Rate limiter ─────────────────────────────────────────────


class _TokenBucket:
    """Simple async token-bucket rate limiter."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._interval <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                wait_time = self._next_allowed - now
            await asyncio.sleep(wait_time)


# ── Pydantic request models ──────────────────────────────────


class _Message(BaseModel):
    role: str
    content: Any  # str | list[dict]


class ChatCompletionRequest(BaseModel):
    model: str = "doubao"
    messages: List[_Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


# ── Application factory ──────────────────────────────────────


def create_app(
    *,
    api_key: Optional[str] = None,
    rpm_limit: float = 20.0,
) -> FastAPI:
    """Build and return a configured FastAPI application."""

    _browser: Dict[str, Any] = {}  # holds BrowserClient instance

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure browser_client logs are visible
        logging.getLogger("doubao2api.browser_client").setLevel(logging.INFO)
        logging.getLogger("doubao2api.browser_client").addHandler(logging.StreamHandler())

        # Start browser client
        headless = os.environ.get("DOUBAO_HEADLESS", "true").lower() == "true"
        user_data_dir = os.environ.get(
            "DOUBAO_BROWSER_DATA",
            os.path.join(os.path.expanduser("~"), ".doubao_browser"),
        )
        client = BrowserClient(headless=headless, user_data_dir=user_data_dir)
        await client.start()
        _browser["client"] = client

        if client.is_ready:
            log.info("Browser client ready (already logged in)")
        else:
            log.warning(
                "Browser not logged in. Visit /auth to scan QR code."
            )

        yield

        # Shutdown
        client = _browser.pop("client", None)
        if client:
            await client.stop()

    app = FastAPI(title="Doubao API", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error", "code": exc.status_code}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exc(request: Request, exc: Exception):
        log.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(exc), "type": "internal_error", "code": 500}},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bucket = _TokenBucket(rpm_limit)

    # ── Auth helper ──

    def _check_auth(request: Request) -> None:
        if not api_key:
            return
        auth = request.headers.get("Authorization", "")
        token = auth[7:].strip() if auth.startswith("Bearer ") else auth.strip()
        if api_key == "any":
            if not token:
                raise HTTPException(status_code=401, detail="API key required")
            return
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _get_client() -> BrowserClient:
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        if not client.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Not logged in. Visit /auth to scan QR code.",
            )
        return client

    # ── Prompt extraction ──

    def _extract_prompt(messages: List[_Message]) -> str:
        """Extract text prompt from OpenAI-format messages."""
        parts: list[str] = []
        for msg in messages:
            if isinstance(msg.content, str):
                if len(messages) == 1:
                    parts.append(msg.content)
                else:
                    parts.append(f"[{msg.role}]: {msg.content}")
            elif isinstance(msg.content, list):
                for p in msg.content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text = p.get("text", "")
                        if text:
                            if len(messages) == 1:
                                parts.append(text)
                            else:
                                parts.append(f"[{msg.role}]: {text}")
        return "\n".join(parts)

    # ── Request logging middleware ──

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        path = request.url.path
        if path.startswith("/auth"):
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000)
        _REQUEST_LOG.append({
            "ts": time.time(),
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "ms": elapsed,
        })
        return response

    # ── Endpoints ──

    @app.get("/health")
    async def health():
        client = _browser.get("client")
        ready = client.is_ready if client else False
        return {"status": "ok" if ready else "not_ready", "logged_in": ready}

    @app.get("/v1/models")
    async def list_models(request: Request):
        _check_auth(request)
        return {"object": "list", "data": ALL_MODELS}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        _check_auth(request)

        use_deep_think = CHAT_MODELS.get(body.model)
        if use_deep_think is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{body.model}'. Available: {', '.join(CHAT_MODELS)}",
            )

        await bucket.acquire()
        client = _get_client()

        prompt = _extract_prompt(body.messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="No text content")

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if body.stream:
            return StreamingResponse(
                _stream_chat(client, prompt, use_deep_think, request_id, body.model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming
        try:
            result = await client.send_message(
                prompt, use_deep_think=use_deep_think
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    async def _stream_chat(
        client: BrowserClient,
        prompt: str,
        use_deep_think: bool,
        request_id: str,
        model: str,
    ):
        """Generate SSE stream in OpenAI format."""
        try:
            result = await client.send_message(
                prompt, use_deep_think=use_deep_think
            )
        except RuntimeError as exc:
            error_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"[Error: {exc}]"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Emit the full text as a single chunk (UI simulation doesn't give real-time streaming)
        text = result["text"]
        if text:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk
        done_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    # ── Auth page (QR login) ──

    @app.get("/auth", response_class=HTMLResponse)
    async def auth_page(request: Request):
        _check_auth(request)
        client = _browser.get("client")
        status = "logged_in" if (client and client.is_ready) else "not_logged_in"
        return _AUTH_HTML.replace("{{STATUS}}", status)

    @app.post("/auth/login")
    async def auth_login(request: Request):
        """Trigger QR login flow. Returns status."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            raise HTTPException(status_code=503, detail="Browser not initialized")
        if client.is_ready:
            return {"status": "already_logged_in"}

        # Start login (non-blocking, returns immediately)
        asyncio.create_task(_do_login(client))
        return {"status": "login_started", "message": "QR code displayed in browser. Scan to login."}

    async def _do_login(client: BrowserClient):
        """Background login task."""
        try:
            ok = await client.wait_for_login(timeout=120)
            if ok:
                log.info("QR login successful via /auth")
            else:
                log.warning("QR login timed out")
        except Exception as exc:
            log.error("QR login error: %s", exc)

    @app.get("/auth/status")
    async def auth_status(request: Request):
        _check_auth(request)
        client = _browser.get("client")
        if client is None:
            return {"logged_in": False, "browser": "not_started"}

        # Do a live check of the page state
        page_url = client.page.url if client.page else ""
        login_btn_count = 0
        if client.page:
            try:
                login_btn = client.page.locator('button:has-text("登录")')
                login_btn_count = await login_btn.count()
            except Exception:
                pass

        actual_logged_in = client.is_ready and login_btn_count == 0

        return {
            "logged_in": actual_logged_in,
            "is_ready_flag": client.is_ready,
            "login_button_visible": login_btn_count > 0,
            "page_url": page_url,
            "device_id": client._device_id or "",
            "web_id": client._web_id or "",
        }

    @app.get("/auth/screenshot")
    async def auth_screenshot(request: Request):
        """Return a screenshot of the browser page (for remote QR viewing)."""
        _check_auth(request)
        client = _browser.get("client")
        if client is None or client.page is None:
            raise HTTPException(status_code=503, detail="Browser not available")
        png_bytes = await client.page.screenshot()
        from fastapi.responses import Response
        return Response(content=png_bytes, media_type="image/png")

    return app


# ── Auth HTML page ──

_AUTH_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Doubao API - Login</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; }
.status { padding: 12px; border-radius: 8px; margin: 20px 0; }
.ok { background: #d4edda; color: #155724; }
.warn { background: #fff3cd; color: #856404; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; margin: 20px 0; }
button { padding: 10px 20px; font-size: 16px; cursor: pointer; border: none; border-radius: 6px; background: #007bff; color: white; }
button:hover { background: #0056b3; }
</style>
</head>
<body>
<h1>Doubao API - Auth</h1>
<div class="status {{STATUS_CLASS}}">
  Status: <strong id="status">{{STATUS}}</strong>
</div>
<div id="actions">
  <button onclick="startLogin()">Start QR Login</button>
  <button onclick="refreshScreenshot()">Refresh Screenshot</button>
</div>
<div id="screenshot-container"></div>
<script>
async function startLogin() {
  const r = await fetch('/auth/login', {method: 'POST', headers: {'Authorization': getAuth()}});
  const d = await r.json();
  alert(d.message || d.status);
  setTimeout(refreshScreenshot, 2000);
}
async function refreshScreenshot() {
  const img = document.createElement('img');
  img.src = '/auth/screenshot?' + Date.now();
  document.getElementById('screenshot-container').innerHTML = '';
  document.getElementById('screenshot-container').appendChild(img);
}
async function checkStatus() {
  const r = await fetch('/auth/status', {headers: {'Authorization': getAuth()}});
  const d = await r.json();
  document.getElementById('status').textContent = d.logged_in ? 'Logged In' : 'Not Logged In';
}
function getAuth() {
  const params = new URLSearchParams(location.search);
  return params.get('key') ? 'Bearer ' + params.get('key') : '';
}
setInterval(checkStatus, 5000);
</script>
</body>
</html>""".replace("{{STATUS_CLASS}}", "ok")


# ── Server runner ──


def run_server():
    """Start the uvicorn server with env-based configuration."""
    import uvicorn

    host = os.environ.get("DOUBAO_HOST", "0.0.0.0")
    port = int(os.environ.get("DOUBAO_PORT", "9090"))
    api_key = os.environ.get("DOUBAO_API_KEY", "")
    rpm = float(os.environ.get("DOUBAO_RPM_LIMIT", "20"))

    app = create_app(api_key=api_key or None, rpm_limit=rpm)

    print(f"\n  Doubao API Server (Playwright)")
    print(f"  Listening on http://{host}:{port}")
    print(f"  Auth page: http://{host}:{port}/auth")
    if api_key:
        print(f"  API Key: {api_key[:4]}{'*' * (len(api_key) - 4)}")
    print()

    uvicorn.run(app, host=host, port=port, log_level="info")
