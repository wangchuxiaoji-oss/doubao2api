"""
Unified multi-modal API server for Doubao.

Exposes all capabilities through OpenAI-compatible endpoints:
  POST /v1/chat/completions     (chat, streaming & non-streaming)
  POST /v1/images/generations   (text-to-image, image-to-image)
  POST /v1/videos/generations   (text-to-video, image-to-video, async)
  GET  /v1/videos/{task_id}     (poll video generation status)
  POST /v1/audio/generations    (music generation)
  GET  /v1/models               (list all available models)

Start with:
    python -m doubao2api
"""
from __future__ import annotations

import asyncio
import base64
import collections
import json
import logging
import os
import platform
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .client import (
    DoubaoChatClient,
    DoubaoChatError,
    DoubaoRateLimitError,
    GeneratedImage,
    GeneratedMusic,
    GeneratedVideo,
    ImageGenerationResult,
    MusicGenerationResult,
    UploadedFile,
    VideoGenerationResult,
)
from .session import load_cookies

log = logging.getLogger("doubao_unified")

# ── Model definitions ────────────────────────────────────────

CHAT_MODELS: Dict[str, int] = {
    "doubao": 0,
    "doubao-quick": 0,
    "doubao-think": 1,
    "doubao-auto": 2,
    "doubao-expert": 3,
    "doubao-pro": 3,
}

IMAGE_MODELS = ["doubao-image"]
VIDEO_MODELS = ["doubao-video"]
AUDIO_MODELS = ["doubao-music"]

ALL_MODELS = (
    [{"id": m, "object": "model", "owned_by": "doubao", "created": 0}
     for m in CHAT_MODELS]
    + [{"id": m, "object": "model", "owned_by": "doubao", "created": 0}
       for m in IMAGE_MODELS]
    + [{"id": m, "object": "model", "owned_by": "doubao", "created": 0}
       for m in VIDEO_MODELS]
    + [{"id": m, "object": "model", "owned_by": "doubao", "created": 0}
       for m in AUDIO_MODELS]
)


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
            # Sleep outside the lock so other coroutines can proceed
            await asyncio.sleep(wait_time)


# ── Pydantic request models ──────────────────────────────────


class _ImageURL(BaseModel):
    url: str
    detail: str = "auto"


class _FileURL(BaseModel):
    url: str  # base64 data URI or HTTP URL


class _ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[_ImageURL] = None
    file_url: Optional[_FileURL] = None


class _Message(BaseModel):
    role: str
    content: Any  # str | list[_ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str = "doubao"
    messages: List[_Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "doubao-image"
    n: int = 1
    size: Optional[str] = None  # ignored, use ratio instead
    ratio: Optional[str] = None  # "1:1", "16:9", "9:16", "4:3", "3:4"
    ref_image_url: Optional[str] = None  # URL or base64 data URI for img2img


class VideoGenerationRequest(BaseModel):
    prompt: str
    model: str = "doubao-video"
    ratio: Optional[str] = None  # "16:9", "9:16", "1:1"
    camera_movement: Optional[str] = None
    ref_image_url: Optional[str] = None  # URL or base64 for img2video
    stream: bool = False  # True = SSE long-poll until done


class AudioGenerationRequest(BaseModel):
    prompt: str
    model: str = "doubao-music"
    lyric: Optional[str] = None
    genre: Optional[str] = None
    mood: Optional[str] = None
    gender: Optional[str] = None
    theme: Optional[str] = None
    generation_type: Optional[str] = None  # "AI_lyric" | "custome_lyric"


# ── Video task store (in-memory) ─────────────────────────────

_video_tasks: Dict[str, Dict[str, Any]] = {}
# Format: {task_id: {"status": "pending"|"processing"|"completed"|"failed",
#                    "result": VideoGenerationResult|None,
#                    "error": str|None, "created": float}}
_VIDEO_TASK_TTL = 3600  # 1 hour — tasks older than this are evicted


def _evict_stale_video_tasks() -> None:
    """Remove video tasks older than TTL to prevent unbounded memory growth."""
    now = time.time()
    stale = [k for k, v in _video_tasks.items() if now - v.get("created", 0) > _VIDEO_TASK_TTL]
    for k in stale:
        del _video_tasks[k]


# ── Application factory ──────────────────────────────────────


def create_app(
    *,
    api_key: Optional[str] = None,
    rpm_limit: float = 50.0,
) -> FastAPI:
    """Build and return a configured FastAPI application with all endpoints."""

    _client_holder: Dict[str, Any] = {}

    async def _session_keepalive_loop():
        """Background task: periodically visit doubao.com/chat to refresh cookies
        via Set-Cookie, then persist the updated cookie jar to disk.

        Doubao API endpoints do NOT return Set-Cookie headers.
        Only page loads (HTML) trigger cookie refresh from the server.
        This mimics what the desktop client does naturally.

        On failure, retries with exponential backoff (30s, 60s, 120s, ...).
        """
        import aiohttp as _aio

        session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
        interval = int(os.environ.get("DOUBAO_KEEPALIVE_INTERVAL", "7200"))
        consecutive_failures = 0
        MAX_RETRY_INTERVAL = 300  # 5 min max between retries

        while True:
            if consecutive_failures > 0:
                # Exponential backoff: 30s, 60s, 120s, 240s, capped at 300s
                retry_wait = min(30 * (2 ** (consecutive_failures - 1)), MAX_RETRY_INTERVAL)
                log.warning(
                    "Keepalive: retry #%d in %ds", consecutive_failures, retry_wait
                )
                await asyncio.sleep(retry_wait)
            else:
                await asyncio.sleep(interval)

            client: Optional[DoubaoChatClient] = _client_holder.get("client")
            if client is None or client._session is None:
                continue
            try:
                # 1. Visit page to trigger Set-Cookie
                async with client._session.get(
                    "https://www.doubao.com/chat",
                    timeout=_aio.ClientTimeout(total=15),
                    allow_redirects=True,
                ) as resp:
                    await resp.read()  # consume body
                    set_cookie_count = sum(
                        1 for k in resp.headers if k.lower() == "set-cookie"
                    )
                    log.info(
                        "Keepalive: GET /chat status=%d, Set-Cookie count=%d",
                        resp.status, set_cookie_count,
                    )
                    if resp.status != 200:
                        raise RuntimeError(f"Keepalive page returned {resp.status}")

                # 2. Persist updated cookies to file
                jar = client._session.cookie_jar
                fresh_cookies: Dict[str, str] = {}
                for cookie in jar:
                    fresh_cookies[cookie.key] = cookie.value

                if not fresh_cookies.get("sessionid"):
                    log.error(
                        "Keepalive: NO sessionid after refresh! "
                        "Session may be dead. Use /v1/session/update to recover."
                    )
                    consecutive_failures += 1
                    continue

                from pathlib import Path
                path = Path(session_file)
                data: Dict[str, Any] = {}
                if path.exists():
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                data["cookies"] = fresh_cookies
                data["sessionid"] = fresh_cookies.get("sessionid", "")
                data["timestamp"] = int(time.time())
                data.setdefault("source", "unknown")
                if "+keepalive" not in data["source"]:
                    data["source"] += "+keepalive"
                path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                log.info("Keepalive: persisted %d cookies", len(fresh_cookies))
                consecutive_failures = 0  # Reset on success
            except Exception as exc:
                consecutive_failures += 1
                log.error(
                    "Keepalive FAILED (#%d): %s", consecutive_failures, exc
                )

    async def _ensure_session_or_login():
        """Check if a valid session exists; if not, print instructions.

        No longer blocks startup or opens local images.
        Users should visit /admin Dashboard to scan QR code.
        """
        session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
        from pathlib import Path

        # Check if session file exists and has cookies
        path = Path(session_file)
        has_session = False
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cookies = data.get("cookies", {})
                if cookies.get("sessionid") and cookies.get("msToken"):
                    has_session = True
            except Exception:
                pass

        # Also check DOUBAO_COOKIE env var
        if not has_session and os.environ.get("DOUBAO_COOKIE", "").strip():
            has_session = True

        if has_session:
            log.info("Session file found, skipping QR login")
            return

        # No valid session — print instructions (non-blocking)
        host = os.environ.get("DOUBAO_HOST", "127.0.0.1")
        port = os.environ.get("DOUBAO_PORT", "9090")
        print()
        print("=" * 60)
        print("  [!] 未检测到有效的 session")
        print()
        print(f"  请访问 Dashboard 扫码登录:")
        print(f"  http://{host}:{port}/admin?key=<YOUR_API_KEY>")
        print()
        print("  或通过 API 手动更新 Cookie:")
        print(f"  POST http://{host}:{port}/v1/session/update")
        print("=" * 60)
        print()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Ensure valid session before starting (interactive QR login if needed)
        await _ensure_session_or_login()
        # If session file exists, eagerly create client so status shows healthy immediately
        session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
        from pathlib import Path
        if Path(session_file).exists():
            try:
                await _create_client()
                log.info("Client initialized from existing session on startup")
            except Exception as exc:
                log.warning("Failed to create client on startup: %s", exc)
        # Start background cookie persistence
        persist_task = asyncio.create_task(_session_keepalive_loop())
        yield
        # Shutdown: cancel background task, persist one last time, close client
        persist_task.cancel()
        client = _client_holder.pop("client", None)
        if client is not None:
            # Final persist before shutdown
            if client._session:
                try:
                    session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
                    jar = client._session.cookie_jar
                    fresh: Dict[str, str] = {c.key: c.value for c in jar}
                    if fresh.get("sessionid"):
                        from pathlib import Path
                        path = Path(session_file)
                        data: Dict[str, Any] = {}
                        if path.exists():
                            try:
                                data = json.loads(path.read_text(encoding="utf-8"))
                            except Exception:
                                pass
                        data["cookies"] = fresh
                        data["sessionid"] = fresh.get("sessionid", "")
                        data["timestamp"] = int(time.time())
                        path.write_text(
                            json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        log.info("Final cookie persist on shutdown")
                except Exception:
                    pass
            await client.__aexit__(None, None, None)

    app = FastAPI(title="Doubao Unified API", version="0.2.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def _http_exception(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.detail, "type": "api_error", "code": exc.status_code}},
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception):
        log.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error", "code": 500}},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ──

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        """Log API requests to the ring buffer (skip /admin paths)."""
        path = request.url.path
        if path.startswith("/admin"):
            return await call_next(request)
        start = time.time()
        response = await call_next(request)
        elapsed = round((time.time() - start) * 1000)  # ms
        model = getattr(request.state, "model", "") if hasattr(request.state, "model") else ""
        _REQUEST_LOG.append({
            "ts": time.time(),
            "method": request.method,
            "path": path,
            "model": model,
            "status": response.status_code,
            "ms": elapsed,
        })
        return response

    bucket = _TokenBucket(rpm_limit)

    # ── Auth helper ──

    def _check_auth(request: Request) -> None:
        if api_key is None or api_key == "":
            return
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
        else:
            token = auth.strip()
        if api_key == "any":
            if not token:
                raise HTTPException(status_code=401, detail="API key required")
            return
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ── Client lifecycle ──

    _client_lock = asyncio.Lock()

    async def _get_client() -> DoubaoChatClient:
        if "client" in _client_holder:
            return _client_holder["client"]
        async with _client_lock:
            # Double-check after acquiring lock
            if "client" in _client_holder:
                return _client_holder["client"]
            return await _create_client()

    async def _create_client() -> DoubaoChatClient:
        """Create a fresh client from session/env config."""
        # Close existing client if any
        old = _client_holder.pop("client", None)
        if old is not None:
            try:
                await old.__aexit__(None, None, None)
            except Exception:
                pass

        ms_token: Optional[str] = os.environ.get("DOUBAO_MS_TOKEN")
        cookies: Optional[Dict[str, str]] = None

        session_file = os.environ.get(
            "DOUBAO_SESSION_FILE", ".doubao_session.json",
        )
        cookies = load_cookies(session_file)

        if not cookies.get("sessionid"):
            raise HTTPException(
                status_code=500,
                detail="No sessionid in session file. Use QR login or /v1/session/update.",
            )

        # Try to get msToken from cookies (optional, API works without it)
        if not ms_token:
            ms_token = cookies.get("msToken", "")

        bot_id = os.environ.get("DOUBAO_BOT_ID", "7338286299411103781")
        timeout = int(os.environ.get("DOUBAO_TIMEOUT", "180"))

        client = DoubaoChatClient(
            cookies=cookies,
            ms_token=ms_token,
            bot_id=bot_id,
            timeout_seconds=timeout,
        )
        await client.__aenter__()
        _client_holder["client"] = client
        _client_holder["created_at"] = time.time()
        log.info("Client created/refreshed successfully")
        return client

    async def _refresh_client() -> DoubaoChatClient:
        """Force-refresh the client after auth failure.

        On headless Linux servers there is no Chromium to pull fresh cookies from.
        The only recovery paths are:
        1. The keepalive loop already refreshed cookies → just reload from file.
        2. User manually updated cookies via /v1/session/update.
        3. User triggered QR login via /v1/session/qr-login.
        """
        log.warning("Refreshing client due to auth failure...")
        return await _create_client()

    def _is_auth_error(exc: Exception) -> bool:
        """Check if an exception indicates an authentication/session failure.

        Uses structured error code matching to avoid false positives.
        Known auth error codes from Doubao:
        - 710012000: "user invalid" (session completely dead)
        - 710012001: "登录已过期，请重新登录" (login expired)
        """
        msg = str(exc)
        # Check for specific Doubao auth error codes
        if "710012" in msg:
            return True
        msg_lower = msg.lower()
        # Check for specific auth-related phrases (not bare numbers)
        auth_phrases = (
            "user invalid", "gateway-error",
            "auth error", "登录已过期",
            "unauthorized", "not_login",
        )
        return any(phrase in msg_lower for phrase in auth_phrases)

    async def _with_auth_retry(coro_factory):
        """Execute an async operation; on auth error, refresh client and retry once."""
        try:
            return await coro_factory()
        except (DoubaoChatError, aiohttp.ClientError, OSError) as exc:
            if not _is_auth_error(exc):
                raise
            # Auth failure — try refresh
            async with _client_lock:
                await _refresh_client()
            return await coro_factory()

    def _fire_and_forget_delete(client: DoubaoChatClient, cid: str) -> None:
        """Schedule conversation deletion without risking unhandled exceptions."""
        async def _safe_delete():
            try:
                await client.delete_conversation(cid)
            except Exception:
                pass  # Best-effort cleanup, already logged inside delete_conversation
        asyncio.ensure_future(_safe_delete())

    # ── Image upload helper ──

    async def _download_and_upload_image(
        client: DoubaoChatClient, url: str,
    ) -> str:
        """Download image from URL/base64 and upload to Doubao, return key."""
        if url.startswith("data:"):
            if "," not in url:
                raise HTTPException(status_code=400, detail="Malformed data URI: missing comma")
            header, b64data = url.split(",", 1)
            img_bytes = base64.b64decode(b64data)
            ext = "png"
            if "jpeg" in header or "jpg" in header:
                ext = "jpeg"
            elif "webp" in header:
                ext = "webp"
            att = await client.upload_image(img_bytes, f"upload.{ext}")
        else:
            import aiohttp as _aio
            timeout = _aio.ClientTimeout(total=30)
            async with _aio.ClientSession(timeout=timeout) as s:
                async with s.get(url) as r:
                    if r.status != 200:
                        raise DoubaoChatError(f"Image fetch failed: HTTP {r.status}")
                    img_bytes = await r.read()
            att = await client.upload_image(img_bytes, "upload.png")
        return att.get("key", "")

    # ── Chat helpers (from server.py) ──

    def _extract_prompt(
        messages: List[_Message],
    ) -> tuple[str, list[_ContentPart], list[_ContentPart]]:
        """Extract text, image parts, and file parts from messages."""
        text_parts: list[str] = []
        image_parts: list[_ContentPart] = []
        file_parts: list[_ContentPart] = []
        for msg in messages:
            role = msg.role
            if isinstance(msg.content, str):
                if len(messages) == 1:
                    text_parts.append(msg.content)
                else:
                    text_parts.append(f"[{role}]: {msg.content}")
            elif isinstance(msg.content, list):
                raw_parts: list[_ContentPart] = []
                for p in msg.content:
                    if isinstance(p, _ContentPart):
                        raw_parts.append(p)
                    elif isinstance(p, dict):
                        try:
                            raw_parts.append(_ContentPart(**p))
                        except Exception:
                            continue
                for part in raw_parts:
                    if part.type == "text" and part.text:
                        if len(messages) == 1:
                            text_parts.append(part.text)
                        else:
                            text_parts.append(f"[{role}]: {part.text}")
                    elif part.type == "image_url":
                        image_parts.append(part)
                    elif part.type == "file_url":
                        file_parts.append(part)
        return "\n".join(text_parts), image_parts, file_parts

    async def _upload_images(
        client: DoubaoChatClient, parts: List[_ContentPart],
    ) -> List[Dict[str, str]]:
        attachments: List[Dict[str, str]] = []
        for part in parts:
            if part.type != "image_url" or part.image_url is None:
                continue
            url = part.image_url.url
            # Already uploaded via /v1/images/upload — extract key directly
            if "tos-cn-i-" in url:
                # CDN URL from previous upload, extract URI key
                from urllib.parse import urlparse as _urlparse
                path = _urlparse(url).path.lstrip("/")
                attachments.append({"uri": path, "cdn_url": url, "name": "upload.png", "format": "png", "width": "64", "height": "64"})
            elif url.startswith("data:"):
                if "," not in url:
                    raise HTTPException(status_code=400, detail="Malformed data URI: missing comma")
                header, b64data = url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                ext = "png"
                if "jpeg" in header or "jpg" in header:
                    ext = "jpeg"
                att = await client.upload_image(img_bytes, f"upload.{ext}")
                attachments.append(att)
            else:
                import aiohttp as _aio
                timeout = _aio.ClientTimeout(total=30)
                async with _aio.ClientSession(timeout=timeout) as s:
                    async with s.get(url) as r:
                        if r.status != 200:
                            raise DoubaoChatError(f"Image fetch: HTTP {r.status}")
                        img_bytes = await r.read()
                att = await client.upload_image(img_bytes, "upload.png")
                attachments.append(att)
        return attachments

    async def _upload_files(
        client: DoubaoChatClient, parts: List[_ContentPart],
    ) -> List[UploadedFile]:
        """Upload file_url content parts and return UploadedFile list."""
        uploaded: List[UploadedFile] = []
        for part in parts:
            if part.type != "file_url" or part.file_url is None:
                continue
            url = part.file_url.url
            if url.startswith("data:"):
                # data:[<mediatype>][;base64],<data>
                if "," not in url:
                    raise HTTPException(status_code=400, detail="Malformed data URI")
                header, b64data = url.split(",", 1)
                file_bytes = base64.b64decode(b64data)
                # Try to extract filename from header or use default
                ext = "txt"
                if "pdf" in header:
                    ext = "pdf"
                elif "csv" in header:
                    ext = "csv"
                filename = f"upload.{ext}"
            else:
                # HTTP(S) URL — download the file
                import aiohttp as _aio
                timeout = _aio.ClientTimeout(total=60)
                async with _aio.ClientSession(timeout=timeout) as s:
                    async with s.get(url) as r:
                        if r.status != 200:
                            raise DoubaoChatError(f"File fetch: HTTP {r.status}")
                        file_bytes = await r.read()
                # Extract filename from URL
                from urllib.parse import urlparse as _urlparse
                path = _urlparse(url).path
                filename = path.rsplit("/", 1)[-1] if "/" in path else "upload.txt"
                if "." not in filename:
                    filename = "upload.txt"
            uf = await client.upload_file(file_bytes, filename)
            uploaded.append(uf)
        return uploaded

    # ── SSE streaming generator for chat ──

    async def _stream_chat(
        client: DoubaoChatClient,
        prompt: str,
        need_deep_think: int,
        image_attachments: Optional[List[Dict[str, str]]],
        file_attachments: Optional[List[UploadedFile]],
        request_id: str,
        model: str,
    ) -> AsyncIterator[str]:
        created = int(time.time())
        conversation_id = ""
        # Track search results by block_id to deduplicate incremental updates.
        # The backend sends the same block_id multiple times with increasing
        # result counts (5→10→15→20). We buffer the latest per block_id and
        # only emit when the block_id changes or non-search content arrives.
        _pending_search_block_id = ""
        _pending_search_data: Optional[Dict[str, Any]] = None
        _search_emitted = False  # True once we've sent at least one search event
        _text_started = False    # True once answer text (delta.content) has begun

        def _flush_search():
            """Emit buffered search result if any."""
            nonlocal _pending_search_block_id, _pending_search_data, _search_emitted
            if _pending_search_data:
                payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "search_results": _pending_search_data,
                        },
                        "finish_reason": None,
                    }],
                }
                _pending_search_data = None
                _pending_search_block_id = ""
                _search_emitted = True
                return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return None

        async for chunk in client.chat_stream_completion(
            text=prompt,
            need_deep_think=need_deep_think,
            image_attachments=image_attachments,
            file_attachments=file_attachments,
        ):
            if chunk.is_done:
                break
            if chunk.conversation_id:
                conversation_id = chunk.conversation_id
                continue

            # Buffer search results, deduplicating by block_id.
            if chunk.search_info and chunk.search_info.get("results"):
                # Once answer text has started AND we already emitted searches
                # during thinking, ignore late "summary" search blocks that the
                # backend sends after the answer — they would confusingly
                # overwrite the per-round results the user already saw.
                if _text_started and _search_emitted:
                    continue
                bid = chunk.search_info.get("block_id", "")
                if bid and bid != _pending_search_block_id:
                    # New search round — flush previous block's final result
                    flushed = _flush_search()
                    if flushed:
                        yield flushed
                # Update buffer with latest (largest) result set for this block
                _pending_search_block_id = bid
                _pending_search_data = chunk.search_info
                continue

            # If we have a buffered search and now see non-search content,
            # flush it before emitting the content.
            if _pending_search_data:
                flushed = _flush_search()
                if flushed:
                    yield flushed

            delta: Dict[str, Any] = {"role": "assistant"}
            if chunk.text:
                delta["content"] = chunk.text
                _text_started = True
            if chunk.thinking:
                delta["reasoning_content"] = chunk.thinking
            if not chunk.text and not chunk.thinking:
                continue
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # Final flush: only if we never emitted search results yet
        # (e.g. non-expert mode where search comes right before text).
        # If we already emitted during thinking, skip — the post-answer
        # "combined" block is redundant and would confuse the client.
        if _pending_search_data and not _search_emitted:
            flushed = _flush_search()
            if flushed:
                yield flushed

        stop_payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(stop_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        # Auto-delete conversation to keep Doubao sidebar clean
        if conversation_id:
            _fire_and_forget_delete(client, conversation_id)

    # ══════════════════════════════════════════════════════════
    # ROUTES
    # ══════════════════════════════════════════════════════════

    @app.get("/health")
    async def health_check():
        return JSONResponse({"status": "ok", "service": "doubao-unified-api"})

    @app.get("/v1/session/status")
    async def session_status(request: Request) -> JSONResponse:
        """Admin endpoint: expose session health for monitoring.

        Query params:
            probe=true  — actually send a test message to verify session (slow, creates conversation)
        By default, only checks cookie presence (fast, no API call).
        """
        _check_auth(request)
        client: Optional[DoubaoChatClient] = _client_holder.get("client")
        if client is None:
            return JSONResponse({
                "status": "no_client",
                "message": "Client not yet initialized",
            })

        created_at = _client_holder.get("created_at", 0)
        age_seconds = int(time.time() - created_at) if created_at else None

        # Check which cookies are present
        cookie_keys = list(client.cookies.keys()) if client.cookies else []
        has_session = "sessionid" in cookie_keys
        has_csrf = (
            "passport_csrf_token" in cookie_keys
            or "passport_csrf_token_default" in cookie_keys
        )
        has_sid_guard = "sid_guard" in cookie_keys

        # Determine health from cookie state (fast, no API call)
        if has_session and has_csrf:
            health = "healthy"
        elif has_session:
            health = "degraded"
        else:
            health = "no_session"

        # Optional live probe (only when explicitly requested)
        do_probe = request.query_params.get("probe", "").lower() in ("true", "1")
        if do_probe:
            try:
                result = await client.chat_completion(
                    text="hi", need_deep_think=0,
                )
                health = "healthy" if result.text else "degraded"
                if result.conversation_id:
                    _fire_and_forget_delete(client, result.conversation_id)
            except Exception as exc:
                msg = str(exc).lower()
                if any(kw in msg for kw in ("710012", "gateway-error", "auth error")):
                    health = "expired"
                else:
                    health = f"error: {str(exc)[:100]}"

        return JSONResponse({
            "status": health,
            "age_seconds": age_seconds,
            "cookies_present": cookie_keys,
            "has_sessionid": has_session,
            "has_csrf_token": has_csrf,
            "has_sid_guard": has_sid_guard,
        })

    @app.post("/v1/session/update")
    async def session_update(request: Request) -> JSONResponse:
        """Update session cookies manually (for headless server recovery).

        Accepts JSON body with either:
        - {"cookies": {"sessionid": "...", ...}}  (dict format)
        - {"cookie_header": "sessionid=xxx; ttwid=yyy; ..."}  (header string)
        """
        _check_auth(request)
        body = await request.json()

        cookies: Dict[str, str] = {}
        if "cookies" in body and isinstance(body["cookies"], dict):
            cookies = {str(k): str(v) for k, v in body["cookies"].items()}
        elif "cookie_header" in body and isinstance(body["cookie_header"], str):
            from .session import _parse_cookie_header
            cookies = _parse_cookie_header(body["cookie_header"])
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide 'cookies' (dict) or 'cookie_header' (string)",
            )

        if not cookies.get("sessionid"):
            raise HTTPException(
                status_code=400,
                detail="cookies must contain 'sessionid'",
            )

        # Persist to session file
        session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
        from pathlib import Path
        path = Path(session_file)
        data: Dict[str, Any] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        data["cookies"] = cookies
        data["sessionid"] = cookies["sessionid"]
        data["timestamp"] = int(time.time())
        data["source"] = "manual_update"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Recreate client with new cookies
        await _create_client()
        log.info("Session updated manually: %d cookies", len(cookies))

        return JSONResponse({
            "status": "ok",
            "message": f"Session updated with {len(cookies)} cookies",
            "cookies_received": list(cookies.keys()),
        })

    # ── QR Login (headless recovery) ──

    _qr_login_state: Dict[str, Any] = {}  # shared state for QR login flow

    @app.post("/v1/session/qr-login")
    async def session_qr_login_start(request: Request) -> JSONResponse:
        """Start QR login flow. Returns base64 QR code PNG for scanning.

        Poll GET /v1/session/qr-login for status updates.
        """
        _check_auth(request)
        from .qr_login import QRLogin, QRStatus

        # Cancel any existing login
        if _qr_login_state.get("instance"):
            _qr_login_state["instance"].cancel()

        qr = QRLogin()
        _qr_login_state.clear()
        _qr_login_state["instance"] = qr
        _qr_login_state["status"] = "starting"
        _qr_login_state["error"] = ""

        import asyncio
        loop = asyncio.get_event_loop()

        def on_status(status: QRStatus, msg: str):
            _qr_login_state["status"] = status.value
            if msg == "qr_ready":
                _qr_login_state["qr_ready"] = True

        def on_done(result):
            if result.status == QRStatus.CONFIRMED:
                _qr_login_state["status"] = "success"
                _qr_login_state["cookies"] = result.cookies
                # Persist and recreate client
                session_file = os.environ.get(
                    "DOUBAO_SESSION_FILE", ".doubao_session.json"
                )
                from pathlib import Path
                data = {
                    "cookies": result.cookies,
                    "sessionid": result.sessionid,
                    "timestamp": int(time.time()),
                    "source": "qr_login",
                }
                Path(session_file).write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                # Schedule client recreation on the event loop
                # Note: must not call _create_client() here (wrong thread);
                # instead schedule it to run on the loop thread.
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(_create_client())
                )
                log.info("QR login success: %d cookies", len(result.cookies))
            else:
                _qr_login_state["status"] = "failed"
                _qr_login_state["error"] = result.error

        qr.start(on_status=on_status, on_done=on_done)

        # Wait briefly for QR code to be generated
        for _ in range(20):  # up to 2 seconds
            await asyncio.sleep(0.1)
            if qr.qrcode_data:
                break

        if qr.qrcode_data:
            import base64 as b64
            qr_b64 = b64.b64encode(qr.qrcode_data).decode()
            return JSONResponse({
                "status": "qr_ready",
                "qr_image_base64": qr_b64,
                "message": "Scan QR code with Doubao mobile app. Poll GET /v1/session/qr-login for status.",
            })
        else:
            return JSONResponse({
                "status": _qr_login_state.get("status", "error"),
                "error": _qr_login_state.get("error", "Failed to generate QR code"),
            }, status_code=502)

    @app.get("/v1/session/qr-login")
    async def session_qr_login_poll(request: Request) -> JSONResponse:
        """Poll QR login status. Returns current state."""
        _check_auth(request)
        status = _qr_login_state.get("status", "idle")
        resp: Dict[str, Any] = {"status": status}

        if status == "success":
            resp["message"] = "Login successful, session updated"
            resp["cookies_count"] = len(_qr_login_state.get("cookies", {}))
        elif status == "failed":
            resp["error"] = _qr_login_state.get("error", "Unknown error")
        elif status == "idle":
            resp["message"] = "No QR login in progress. POST /v1/session/qr-login to start."

        return JSONResponse(resp)

    @app.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        _check_auth(request)
        return JSONResponse({"object": "list", "data": ALL_MODELS})

    # ── POST /v1/files (upload file for chat attachment) ──

    @app.post("/v1/files")
    async def upload_file_endpoint(request: Request):
        """Upload a file for use as a chat attachment.

        Accepts multipart/form-data with a 'file' field.
        Returns an object with the file URI that can be used in chat messages
        via the file_url content part type.

        Example response:
            {"id": "file-xxx", "object": "file", "filename": "doc.pdf",
             "bytes": 1234, "uri": "tos-cn-i-xxx/yyy.pdf"}
        """
        _check_auth(request)
        content_type = request.headers.get("content-type", "")
        if "multipart" not in content_type:
            raise HTTPException(
                status_code=400,
                detail="Expected multipart/form-data with a 'file' field",
            )

        form = await request.form()
        file_field = form.get("file")
        if file_field is None:
            raise HTTPException(status_code=400, detail="Missing 'file' field")

        file_bytes = await file_field.read()
        filename = getattr(file_field, "filename", None) or "upload.txt"

        await bucket.acquire()
        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            uploaded = await client.upload_file(file_bytes, filename)
        except DoubaoChatError as exc:
            raise HTTPException(status_code=502, detail=f"File upload: {exc}")

        return JSONResponse({
            "id": f"file-{uuid.uuid4().hex[:24]}",
            "object": "file",
            "filename": uploaded.name,
            "bytes": uploaded.size,
            "uri": uploaded.uri,
            "file_type": uploaded.file_type,
            "purpose": "assistants",
        })

    # ── POST /v1/images/upload (upload image for chat, returns usable URL) ──

    @app.post("/v1/images/upload")
    async def upload_image_endpoint(request: Request):
        """Upload an image via multipart/form-data for use in chat.

        Returns a URL that can be directly used in image_url content parts,
        avoiding the need to base64-encode large images.

        Example:
            curl -F "file=@photo.jpg" http://host:9090/v1/images/upload
            -> {"url": "https://...", "key": "tos-cn-i-.../xxx.png"}

        Then use in chat:
            {"type": "image_url", "image_url": {"url": "<returned url>"}}
        """
        _check_auth(request)
        content_type = request.headers.get("content-type", "")
        if "multipart" not in content_type:
            raise HTTPException(
                status_code=400,
                detail="Expected multipart/form-data with a 'file' field",
            )

        form = await request.form()
        file_field = form.get("file")
        if file_field is None:
            raise HTTPException(status_code=400, detail="Missing 'file' field")

        img_bytes = await file_field.read()
        filename = getattr(file_field, "filename", None) or "upload.png"

        await bucket.acquire()
        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        try:
            att = await client.upload_image(img_bytes, filename)
        except DoubaoChatError as exc:
            raise HTTPException(status_code=502, detail=f"Image upload: {exc}")

        return JSONResponse({
            "url": att.get("cdn_url", ""),
            "key": att.get("uri", ""),
            "filename": filename,
            "bytes": len(img_bytes),
        })

    # ── POST /v1/chat/completions ──

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        _check_auth(request)
        request.state.model = body.model
        need_deep_think = CHAT_MODELS.get(body.model)
        if need_deep_think is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown chat model '{body.model}'. "
                       f"Available: {', '.join(CHAT_MODELS)}",
            )
        await bucket.acquire()
        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        prompt, image_parts, file_parts = _extract_prompt(body.messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="No text content")

        image_attachments: Optional[List[Dict[str, str]]] = None
        if image_parts:
            try:
                image_attachments = await _upload_images(client, image_parts)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"Image upload: {exc}")

        file_attachments: Optional[List[UploadedFile]] = None
        if file_parts:
            try:
                file_attachments = await _upload_files(client, file_parts)
            except Exception as exc:
                raise HTTPException(status_code=502, detail=f"File upload: {exc}")

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if body.stream:
            # Streaming with auth retry on first chunk
            async def _start_stream():
                c = await _get_client()
                g = _stream_chat(
                    c, prompt, need_deep_think,
                    image_attachments, file_attachments, request_id, body.model,
                )
                first = await g.__anext__()
                return first, g

            try:
                first_chunk, gen = await _start_stream()
            except (DoubaoChatError, StopAsyncIteration) as exc:
                if isinstance(exc, StopAsyncIteration):
                    first_chunk, gen = None, None
                elif _is_auth_error(exc):
                    await _refresh_client()
                    try:
                        first_chunk, gen = await _start_stream()
                    except StopAsyncIteration:
                        first_chunk, gen = None, None
                    except DoubaoRateLimitError as exc2:
                        raise HTTPException(status_code=429, detail=str(exc2))
                    except DoubaoChatError as exc2:
                        raise HTTPException(status_code=502, detail=str(exc2))
                else:
                    raise HTTPException(status_code=502, detail=str(exc))
            except DoubaoRateLimitError as exc:
                raise HTTPException(status_code=429, detail=str(exc))

            async def _stream_with_first():
                if first_chunk:
                    yield first_chunk
                if gen:
                    async for part in gen:
                        yield part

            return StreamingResponse(
                _stream_with_first(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming (with auth retry)
        async def _do_chat():
            c = await _get_client()
            r = await c.chat_completion(
                text=prompt,
                need_deep_think=need_deep_think,
                image_attachments=image_attachments,
                file_attachments=file_attachments,
            )
            return r

        try:
            result = await _with_auth_retry(_do_chat)
        except DoubaoRateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except DoubaoChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        # Auto-delete conversation
        if result.conversation_id:
            client = _client_holder.get("client")
            if client:
                _fire_and_forget_delete(client, result.conversation_id)

        message: Dict[str, Any] = {"role": "assistant", "content": result.text}
        if result.thinking_text:
            message["reasoning_content"] = result.thinking_text

        response_body: Dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Include search results if present (only final results with data)
        final_searches = [
            s for s in result.search_results if s.get("results")
        ]
        if final_searches:
            response_body["search_results"] = final_searches[-1]

        return JSONResponse(response_body)

    # ── POST /v1/images/generations ──

    @app.post("/v1/images/generations")
    async def images_generations(body: ImageGenerationRequest, request: Request):
        _check_auth(request)
        request.state.model = body.model
        await bucket.acquire()

        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # Handle ref_image upload for img2img
        ref_image_key: Optional[str] = None
        if body.ref_image_url:
            try:
                ref_image_key = await _download_and_upload_image(
                    client, body.ref_image_url,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"Ref image upload failed: {exc}",
                )

        async def _do_image():
            c = await _get_client()
            return await c.generate_image(
                prompt=body.prompt,
                ratio=body.ratio,
                ref_image_key=ref_image_key,
            )

        try:
            result = await _with_auth_retry(_do_image)
        except DoubaoRateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except DoubaoChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        # OpenAI-compatible response format
        data = []
        for img in result.images:
            item: Dict[str, Any] = {"url": img.ori_url or img.raw_url or img.thumb_url}
            if img.width:
                item["width"] = img.width
            if img.height:
                item["height"] = img.height
            if img.raw_url:
                item["raw_url"] = img.raw_url
            data.append(item)

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })

    # ── POST /v1/videos/generations ──

    @app.post("/v1/videos/generations")
    async def videos_generations(body: VideoGenerationRequest, request: Request):
        _check_auth(request)
        request.state.model = body.model
        await bucket.acquire()

        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        # Handle ref_image upload for img2video
        ref_image_key: Optional[str] = None
        if body.ref_image_url:
            try:
                ref_image_key = await _download_and_upload_image(
                    client, body.ref_image_url,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=502, detail=f"Ref image upload failed: {exc}",
                )

        if body.stream:
            # SSE mode: long-poll until video is ready
            async def _video_stream() -> AsyncIterator[str]:
                try:
                    result = await client.generate_video(
                        prompt=body.prompt,
                        ratio=body.ratio,
                        camera_movement=body.camera_movement,
                        ref_image_key=ref_image_key,
                        timeout=300,
                    )
                    data = []
                    for v in result.videos:
                        item: Dict[str, Any] = {"url": v.video_url}
                        if v.cover_url:
                            item["cover_url"] = v.cover_url
                        if v.duration:
                            item["duration"] = v.duration
                        data.append(item)
                    payload = {
                        "status": "completed",
                        "created": int(time.time()),
                        "data": data,
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except DoubaoChatError as exc:
                    err = {"status": "failed", "error": str(exc)}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                _video_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-stream: submit task, return task_id for polling
        task_id = f"vtask-{uuid.uuid4().hex[:16]}"
        _evict_stale_video_tasks()
        _video_tasks[task_id] = {
            "status": "processing",
            "result": None,
            "error": None,
            "created": time.time(),
        }

        # Launch background task
        async def _run_video():
            try:
                result = await client.generate_video(
                    prompt=body.prompt,
                    ratio=body.ratio,
                    camera_movement=body.camera_movement,
                    ref_image_key=ref_image_key,
                    timeout=300,
                )
                _video_tasks[task_id]["status"] = "completed"
                _video_tasks[task_id]["result"] = result
            except Exception as exc:
                _video_tasks[task_id]["status"] = "failed"
                _video_tasks[task_id]["error"] = str(exc)

        asyncio.create_task(_run_video())

        return JSONResponse(
            status_code=202,
            content={
                "id": task_id,
                "status": "processing",
                "created": int(time.time()),
            },
        )

    # ── GET /v1/videos/{task_id} ──

    @app.get("/v1/videos/{task_id}")
    async def get_video_task(task_id: str, request: Request):
        _check_auth(request)

        task = _video_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        response: Dict[str, Any] = {
            "id": task_id,
            "status": task["status"],
            "created": int(task["created"]),
        }

        if task["status"] == "completed" and task["result"]:
            result: VideoGenerationResult = task["result"]
            data = []
            for v in result.videos:
                item: Dict[str, Any] = {"url": v.video_url}
                if v.cover_url:
                    item["cover_url"] = v.cover_url
                if v.duration:
                    item["duration"] = v.duration
                if v.width:
                    item["width"] = v.width
                if v.height:
                    item["height"] = v.height
                data.append(item)
            response["data"] = data
        elif task["status"] == "failed":
            response["error"] = task["error"]

        return JSONResponse(response)

    # ── POST /v1/audio/generations ──

    @app.post("/v1/audio/generations")
    async def audio_generations(body: AudioGenerationRequest, request: Request):
        _check_auth(request)
        request.state.model = body.model
        await bucket.acquire()

        try:
            client = await _get_client()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        async def _do_music():
            c = await _get_client()
            return await c.generate_music(
                prompt=body.prompt,
                lyric=body.lyric,
                genre=body.genre,
                mood=body.mood,
                gender=body.gender,
                theme=body.theme,
                generation_type=body.generation_type,
            )

        try:
            result = await _with_auth_retry(_do_music)
        except DoubaoRateLimitError as exc:
            raise HTTPException(status_code=429, detail=str(exc))
        except DoubaoChatError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        data = []
        for track in result.tracks:
            item: Dict[str, Any] = {"url": track.audio_url}
            if track.title:
                item["title"] = track.title
            if track.duration:
                item["duration"] = track.duration
            if track.lyrics:
                item["lyrics"] = track.lyrics
            if track.cover_url:
                item["cover_url"] = track.cover_url
            data.append(item)

        return JSONResponse({
            "created": int(time.time()),
            "data": data,
        })

    # ══════════════════════════════════════════════════════════
    # ADMIN DASHBOARD ROUTES
    # ══════════════════════════════════════════════════════════

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_dashboard(request: Request):
        """Serve the admin dashboard HTML (no auth — it's just a static shell)."""
        html_path = Path(__file__).parent / "static" / "admin.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="admin.html not found")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/admin/api/logs")
    async def admin_logs(request: Request):
        """Return recent request logs from ring buffer."""
        _check_auth(request)
        return JSONResponse(list(_REQUEST_LOG))

    @app.get("/admin/api/system")
    async def admin_system(request: Request):
        """Return system information."""
        _check_auth(request)
        uptime = int(time.time() - _SERVER_START_TIME)
        return JSONResponse({
            "python_version": sys.version,
            "platform": platform.platform(),
            "uptime_seconds": uptime,
            "rpm_limit": rpm_limit,
            "api_key_mode": (
                "strict" if api_key and api_key != "any"
                else "accept-any" if api_key == "any"
                else "disabled"
            ),
            "keepalive_interval": int(os.environ.get(
                "DOUBAO_KEEPALIVE_INTERVAL", "7200"
            )),
            "host": os.environ.get("DOUBAO_HOST", "127.0.0.1"),
            "port": int(os.environ.get("DOUBAO_PORT", "9090")),
            "session_file": os.environ.get(
                "DOUBAO_SESSION_FILE", ".doubao_session.json"
            ),
            "models": {
                "chat": list(CHAT_MODELS.keys()),
                "image": IMAGE_MODELS,
                "video": VIDEO_MODELS,
                "audio": AUDIO_MODELS,
            },
        })

    @app.get("/admin/api/cookies")
    async def admin_cookies(request: Request):
        """Return current session cookies with masked values."""
        _check_auth(request)
        client: Optional[DoubaoChatClient] = _client_holder.get("client")
        if client is None:
            return JSONResponse({"cookies": [], "session_file": None})

        session_file = os.environ.get("DOUBAO_SESSION_FILE", ".doubao_session.json")
        # Read session file metadata
        file_info = None
        try:
            from pathlib import Path
            p = Path(session_file)
            if p.exists():
                import json as _json
                data = _json.loads(p.read_text(encoding="utf-8"))
                file_info = {
                    "path": str(p.resolve()),
                    "source": data.get("source", "unknown"),
                    "timestamp": data.get("timestamp"),
                }
        except Exception:
            pass

        # Build cookie list
        cookies_list = []
        if client.cookies:
            for k, v in client.cookies.items():
                val = str(v)
                cookies_list.append({
                    "name": k,
                    "value": val,
                    "length": len(val),
                })

        return JSONResponse({
            "cookies": cookies_list,
            "total": len(cookies_list),
            "session_file": file_info,
        })

    return app


# ── Convenience entry point ──────────────────────────────────


def run_server() -> None:
    """Start the unified API server with settings from environment variables."""
    import uvicorn

    host = os.environ.get("DOUBAO_HOST", "127.0.0.1")
    port = int(os.environ.get("DOUBAO_PORT", "9090"))
    api_key = os.environ.get("DOUBAO_API_KEY", "")
    rpm = float(os.environ.get("DOUBAO_RPM_LIMIT", "50"))
    log_level = os.environ.get("DOUBAO_LOG_LEVEL", "info")

    app = create_app(api_key=api_key or None, rpm_limit=rpm)

    print(f"Doubao Unified API starting on http://{host}:{port}")
    print(f"  Rate limit : {rpm} rpm")
    auth_status = (
        "strict" if api_key and api_key != "any"
        else "accept-any" if api_key == "any"
        else "disabled"
    )
    print(f"  API key    : {auth_status}")
    print()
    print("Endpoints:")
    print(f"  GET  /v1/models")
    print(f"  POST /v1/files")
    print(f"  POST /v1/chat/completions")
    print(f"  POST /v1/images/generations")
    print(f"  POST /v1/videos/generations")
    print(f"  GET  /v1/videos/{{task_id}}")
    print(f"  POST /v1/audio/generations")
    print(f"  GET  /admin                  (Dashboard)")
    print()
    print("Chat models:", ", ".join(CHAT_MODELS))
    print("Image model:", ", ".join(IMAGE_MODELS))
    print("Video model:", ", ".join(VIDEO_MODELS))
    print("Audio model:", ", ".join(AUDIO_MODELS))

    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    run_server()
