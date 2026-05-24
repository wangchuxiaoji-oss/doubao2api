"""Playwright-based Qianwen (通义千问) client with in-browser fetch.

Architecture:
- Playwright: Maintains a browser session on qianwen.com
- In-browser fetch(): Uses page's security SDK (qwenSign + etSign + baxia)
  to auto-generate all required security headers
- expose_function bridge: Streams SSE chunks from browser JS back to Python

Key signing flow (all done in-browser):
  1. qwenSign(url) → signedHeader + signedUrl
  2. etSign(signedUrl) → bx_et header
  3. postFYModule.getFYToken() → bx-ua header
  4. postFYModule.getUidToken() → bx-umidtoken header

No login required for basic usage (temporary sessions).
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any

from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import Stealth

log = logging.getLogger(__name__)

QIANWEN_URL = "https://www.qianwen.com"
CHAT_URL = f"{QIANWEN_URL}/chat/"
CHAT_API_URL = "https://chat2.qianwen.com/api/v2/chat"

# Model mapping for Qianwen
# modelCode values from https://chat2-api.qianwen.com/api/v1/model/list
QIANWEN_MODELS = {
    "qianwen": {"model": "Qwen", "deep_search": "0"},
    "qianwen-max": {"model": "Qwen3.7-Max", "deep_search": "0"},
    "qianwen-max-think": {"model": "Qwen3.7-Max", "deep_search": "1"},
    "qianwen-think": {"model": "Qwen3-Max-Thinking-Preview", "deep_search": "0"},
    "qianwen-coder": {"model": "Qwen3-Coder", "deep_search": "0"},
    "qianwen-flash": {"model": "Qwen3.5-Flash", "deep_search": "0"},
    "qianwen-search": {"model": "Qwen3.7-Max", "deep_search": "1"},
    # Direct model codes (pass-through)
    "Qwen": {"model": "Qwen", "deep_search": "0"},
    "Qwen3.7-Max": {"model": "Qwen3.7-Max", "deep_search": "0"},
    "Qwen3.5-Plus": {"model": "Qwen3.5-Plus", "deep_search": "0"},
    "Qwen3.5-Flash": {"model": "Qwen3.5-Flash", "deep_search": "0"},
    "Qwen3-Max": {"model": "Qwen3-Max", "deep_search": "0"},
    "Qwen3-Max-Thinking-Preview": {"model": "Qwen3-Max-Thinking-Preview", "deep_search": "0"},
    "Qwen3-Coder": {"model": "Qwen3-Coder", "deep_search": "0"},
    "Qwen3-Coder-Flash": {"model": "Qwen3-Coder-Flash", "deep_search": "0"},
    "Qwen3-235B-A22B-2507": {"model": "Qwen3-235B-A22B-2507", "deep_search": "0"},
}


class QianwenClient:
    """Manages Playwright for Qianwen in-browser fetch API calls."""

    def __init__(self, headless: bool = True, user_data_dir: Optional[str] = None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._ready = False
        self._device_id: str = str(uuid.uuid4())
        # Stream bridge: request_id -> asyncio.Queue for SSE chunks
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._bridge_ready: bool = False
        # Session management
        self._session_id: Optional[str] = None
        self._topic_id: Optional[str] = None
        self._consecutive_failures: int = 0

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def record_success(self):
        self._consecutive_failures = 0

    def record_failure(self):
        self._consecutive_failures += 1

    async def start(self):
        """Launch browser and navigate to Qianwen."""
        log.info("Starting Qianwen browser client (headless=%s)", self.headless)
        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ]

        if self.user_data_dir:
            self._context = await self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                args=launch_args,
                viewport={"width": 1280, "height": 800},
                locale="zh-CN",
            )
            self._page = (
                self._context.pages[0]
                if self._context.pages
                else await self._context.new_page()
            )
        else:
            browser = await self._playwright.chromium.launch(
                headless=self.headless, args=launch_args
            )
            self._context = await browser.new_context(
                viewport={"width": 1280, "height": 800}, locale="zh-CN"
            )
            self._page = await self._context.new_page()

        # Apply stealth
        stealth = Stealth(navigator_languages_override=("zh-CN", "zh"))
        await stealth.apply_stealth_async(self._page)

        # Navigate to Qianwen
        await self._page.goto(CHAT_URL, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(3)

        # Wait for security SDK to load
        await self._page.wait_for_function(
            "() => window.__QIANWEN_CHAT_SDK__ && window.etSign && window.__baxia__",
            timeout=30000,
        )
        log.info("Qianwen security SDK loaded")

        # Setup stream bridge
        await self._setup_stream_bridge()
        self._ready = True
        log.info("Qianwen client ready")

    async def stop(self):
        """Close browser."""
        self._ready = False
        if self._context:
            await self._context.close()
        if self._playwright:
            await self._playwright.stop()
        log.info("Qianwen client stopped")

    async def _setup_stream_bridge(self):
        """Expose Python callback to browser for streaming SSE chunks."""
        if self._bridge_ready:
            return

        async def on_stream_chunk(req_id: str, chunk_json: str):
            """Called from browser JS for each SSE data line."""
            q = self._stream_queues.get(req_id)
            if q:
                await q.put(chunk_json)

        await self._page.expose_function("__qwStreamChunk", on_stream_chunk)
        self._bridge_ready = True
        log.debug("Stream bridge function exposed")

    # ------------------------------------------------------------------
    # In-browser fetch with SSE streaming
    # ------------------------------------------------------------------

    _FETCH_JS = """
    async ([reqId, sessId, topicId, content, model, deepSearch, deviceId]) => {
        try {
            const qwenSign = window.__QIANWEN_CHAT_SDK__.qwenSign;
            const baxia = window.__baxia__;

            const timestamp = Date.now();
            const nonce = Math.random().toString(36).substring(2, 13);
            const baseUrl = `https://chat2.qianwen.com/api/v2/chat?biz_id=ai_qwen&fe_version=1.0.0&chat_client=h5&device=pc&fr=pc&pr=qwen&ut=${deviceId}&la=zh-CN&tz=Pacific%2FAuckland&wv=2.9.3&ve=2.9.3&nonce=${nonce}&timestamp=${timestamp}`;

            const { signedHeader, signedUrl } = await qwenSign(baseUrl);
            const bxEt = window.etSign(signedUrl);
            const bxUa = baxia.postFYModule.getFYToken();
            const bxUmid = baxia.postFYModule.getUidToken();

            const body = {
                req_id: reqId,
                parent_req_id: "",
                messages: [{
                    mime_type: "text/plain",
                    content: content,
                    meta_data: { ori_query: content },
                    status: "complete"
                }],
                scene: "chat",
                sub_scene: "",
                scene_param: topicId ? "continue_chat" : "new_chat",
                session_id: sessId,
                biz_id: "ai_qwen",
                topic_id: topicId || "",
                model: model,
                from: "default",
                protocol_version: "v2",
                messages_merge: false,
                chat_client: "h5",
                deep_search: deepSearch,
                temporary: !topicId
            };

            const headers = {
                'Content-Type': 'application/json',
                'accept': 'application/json, text/event-stream, text/plain, */*',
                'x-platform': 'pc_tongyi',
                'x-device-id': deviceId,
                'x-chat-id': reqId,
                'x-chat-biz': JSON.stringify({chatId: reqId, agentId: "", enableWebp: ""}),
                'bx_et': bxEt,
                'bx-ua': bxUa,
                'bx-umidtoken': bxUmid,
                ...signedHeader
            };

            const resp = await fetch(signedUrl, {
                method: 'POST',
                headers,
                body: JSON.stringify(body),
                credentials: 'include'
            });

            if (!resp.ok) {
                const errText = await resp.text();
                await window.__qwStreamChunk(reqId,
                    '__HTTP_ERROR__:' + resp.status + ':' + errText.slice(0, 500));
                await window.__qwStreamChunk(reqId, null);
                return;
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const dataStr = line.substring(5);
                        if (dataStr && dataStr !== '{}') {
                            await window.__qwStreamChunk(reqId, dataStr);
                        }
                    }
                    // Skip 'event:' lines (Qwen3.7-Max uses event:message prefix)
                }
            }
            // Remaining buffer
            if (buffer.startsWith('data:')) {
                const dataStr = buffer.substring(5);
                if (dataStr && dataStr !== '{}') {
                    await window.__qwStreamChunk(reqId, dataStr);
                }
            }
            // Signal end
            await window.__qwStreamChunk(reqId, null);
        } catch(e) {
            await window.__qwStreamChunk(reqId, '__ERROR__:' + e.message);
            await window.__qwStreamChunk(reqId, null);
        }
    }
    """

    async def _browser_fetch_stream(
        self, req_id: str, sess_id: str, topic_id: str,
        content: str, model: str, deep_search: str,
    ):
        """Execute fetch() inside browser and stream via callback."""
        await self._page.evaluate(
            self._FETCH_JS,
            [req_id, sess_id, topic_id, content, model, deep_search, self._device_id],
        )

    # ------------------------------------------------------------------
    # Public streaming chat API
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list,
        model: str = "Qwen",
        deep_search: str = "0",
        session_id: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and yield parsed SSE chunks.

        Args:
            messages: List of dicts with 'role' and 'content'
            model: Qianwen model name (default "Qwen")
            deep_search: "0" normal, "1" deep search
            session_id: Reuse session for multi-turn
            topic_id: Reuse topic for multi-turn

        Yields:
            Parsed JSON dicts from each SSE data line
        """
        if not self._ready:
            raise RuntimeError("Qianwen client not ready, call start() first")

        req_id = uuid.uuid4().hex
        sess_id = session_id or uuid.uuid4().hex

        # Extract last user message
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg["content"]
                break
        if not last_user_msg:
            raise ValueError("No user message found in messages")

        # Create queue and launch fetch
        queue: asyncio.Queue = asyncio.Queue()
        self._stream_queues[req_id] = queue

        log.info("Qianwen chat: model=%s deep_search=%s len=%d",
                 model, deep_search, len(last_user_msg))

        eval_task = asyncio.create_task(
            self._browser_fetch_stream(
                req_id, sess_id, topic_id or "", last_user_msg, model, deep_search
            )
        )

        try:
            while True:
                chunk = await asyncio.wait_for(queue.get(), timeout=120)
                if chunk is None:
                    break
                if chunk.startswith("__ERROR__:"):
                    error_msg = chunk[10:]
                    log.error("Browser fetch error: %s", error_msg[:200])
                    self.record_failure()
                    yield {"error": True, "message": error_msg}
                    break
                if chunk.startswith("__HTTP_ERROR__:"):
                    parts = chunk[15:].split(":", 1)
                    status = int(parts[0]) if parts[0].isdigit() else 0
                    body = parts[1] if len(parts) > 1 else ""
                    log.error("Qianwen API error %d: %s", status, body[:200])
                    self.record_failure()
                    yield {"error": True, "status": status, "message": body}
                    break
                # Parse JSON
                try:
                    data = json.loads(chunk)
                    if isinstance(data, dict):
                        yield data
                    # Skip non-dict values (e.g., bare "true" from event:complete)
                except json.JSONDecodeError:
                    continue
        except asyncio.TimeoutError:
            log.error("Stream timeout (120s) for request %s", req_id)
            self.record_failure()
            yield {"error": True, "message": "Stream timeout"}
        finally:
            self._stream_queues.pop(req_id, None)
            if not eval_task.done():
                eval_task.cancel()
            else:
                try:
                    eval_task.result()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list,
        model: str = "Qwen",
        deep_search: str = "0",
    ) -> Dict[str, Any]:
        """Send message, collect full response.

        Returns:
            Dict with 'content', 'think_content', 'model', 'usage' keys
        """
        full_content = ""
        full_think = ""
        usage = {}
        actual_model = ""

        async for chunk in self.chat_stream(messages, model, deep_search):
            if chunk.get("error"):
                raise RuntimeError(chunk.get("message", "Unknown error"))

            data = chunk.get("data", {})
            msgs = data.get("messages", [])
            for msg in msgs:
                if msg.get("mime_type") == "multi_load/iframe" and msg.get("content"):
                    full_content = msg["content"]  # cumulative
                    # Extract think_content from meta_data
                    meta = msg.get("meta_data", {})
                    multi_load = meta.get("multi_load", [])
                    if multi_load and isinstance(multi_load, list):
                        ml_content = multi_load[0].get("content", {})
                        if isinstance(ml_content, dict):
                            tc = ml_content.get("think_content", "")
                            if tc:
                                full_think = tc

            # Extract usage from final chunks
            extra = data.get("extra_info", {})
            chat_odps = extra.get("chat_odps", {})
            if chat_odps.get("total_usage"):
                usage = chat_odps["total_usage"]
            if chat_odps.get("model_info", {}).get("model"):
                actual_model = chat_odps["model_info"]["model"]

        self.record_success()
        return {
            "content": full_content,
            "think_content": full_think,
            "model": actual_model or model,
            "usage": usage,
        }
