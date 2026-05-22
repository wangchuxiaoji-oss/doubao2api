"""Playwright-based Doubao browser client.

Uses a real Chromium browser to interact with doubao.com.
ByteDance's frontend JS automatically handles:
- a_bogus signature injection
- msToken management
- mssdk device fingerprint registration
"""

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Optional, Dict, Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth

log = logging.getLogger(__name__)

DOUBAO_URL = "https://www.doubao.com"
CHAT_URL = f"{DOUBAO_URL}/chat/"


class BrowserClient:
    """Manages a Playwright browser session for Doubao API calls."""

    def __init__(self, headless: bool = True, user_data_dir: Optional[str] = None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._ready = False
        self._bot_id: Optional[str] = None
        self._device_id: Optional[str] = None
        self._web_id: Optional[str] = None
        self._fp: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def page(self) -> Optional[Page]:
        return self._page

    async def start(self):
        """Launch browser and navigate to Doubao."""
        log.info("Starting Playwright browser (headless=%s)", self.headless)
        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        if self.user_data_dir:
            # Persistent context preserves cookies across restarts
            self._context = await self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                args=launch_args,
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                args=launch_args,
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
            )
            self._page = await self._context.new_page()

        # Apply stealth patches
        stealth = Stealth(navigator_languages_override=("zh-CN", "zh"))
        await stealth.apply_stealth_async(self._page)
        log.info("Browser started, navigating to Doubao...")

        await self._page.goto(CHAT_URL, wait_until="domcontentloaded", timeout=30000)
        await self._check_login_state()

    async def _check_login_state(self):
        """Check if we're logged in or need QR scan."""
        url = self._page.url
        if "security/doubao-region-ban" in url:
            log.info("Not logged in - need QR scan login")
            self._ready = False
            return

        # Check if chat page loaded (has input box)
        try:
            await self._page.wait_for_selector(
                'textarea[placeholder*="发消息"]', timeout=10000
            )
            self._ready = True
            await self._extract_params()
            log.info("Logged in and ready! bot_id=%s", self._bot_id)
        except Exception:
            log.warning("Page loaded but chat input not found")
            self._ready = False

    async def _extract_params(self):
        """Extract device_id, web_id, bot_id from page (retries for async localStorage population)."""
        for _ in range(5):
            params = await self._page.evaluate("""() => {
                const result = {};
                // Get device_id from localStorage
                try {
                    const samWeb = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');
                    result.device_id = samWeb.web_id || '';
                } catch(e) {}
                // Get web_id/tea_uuid from localStorage
                try {
                    const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
                    result.web_id = tea.web_id || '';
                } catch(e) {}
                // Get fp from cookie
                const fpCookie = document.cookie.split(';')
                    .map(c => c.trim())
                    .find(c => c.startsWith('s_v_web_id='));
                result.fp = fpCookie ? fpCookie.split('=')[1] : '';
                return result;
            }""")
            self._device_id = params.get("device_id", "")
            self._web_id = params.get("web_id", "")
            self._fp = params.get("fp", "")
            if self._device_id and self._web_id:
                break
            await asyncio.sleep(1)
        log.info("Params: device_id=%s, web_id=%s, fp=%s", self._device_id, self._web_id, self._fp[:20] if self._fp else "")

    async def get_qr_code_screenshot(self) -> Optional[bytes]:
        """Click login and return QR code screenshot as PNG bytes."""
        if self._ready:
            return None

        url = self._page.url
        if "security/doubao-region-ban" in url:
            # Click login button
            btn = self._page.locator('button:has-text("登录")')
            if await btn.count() > 0:
                await btn.click()
                await asyncio.sleep(2)

        # Try to find QR code image
        qr = self._page.locator('img[src*="qrcode"], canvas').first
        try:
            await qr.wait_for(timeout=5000)
            return await qr.screenshot()
        except Exception:
            # Fallback: screenshot the login dialog area
            return await self._page.screenshot()

    async def wait_for_login(self, timeout: int = 120) -> bool:
        """Wait for user to scan QR code and complete login."""
        # First, ensure login dialog is showing
        await self._trigger_login_dialog()

        log.info("Waiting for QR scan login (timeout=%ds)...", timeout)
        try:
            await self._page.wait_for_url(
                "**/chat/**", timeout=timeout * 1000
            )
            await self._page.wait_for_selector(
                'textarea[placeholder*="发消息"]', timeout=15000
            )
            self._ready = True
            await self._extract_params()
            log.info("Login successful!")
            return True
        except Exception as e:
            log.error("Login timeout: %s", e)
            return False

    async def _trigger_login_dialog(self):
        """Click login button if on security/ban page."""
        url = self._page.url
        if "security/doubao-region-ban" in url:
            btn = self._page.locator('button:has-text("登录")')
            if await btn.count() > 0:
                await btn.click()
                log.info("Clicked login button, QR code should appear")
                await asyncio.sleep(2)

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and yield SSE events.

        The fetch is executed inside the browser, so ByteDance's JS
        automatically injects a_bogus and msToken.
        """
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        need_create = conversation_id is None
        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}" if need_create else ""
        effective_bot_id = bot_id or "7338286299411103781"

        js_code = """
        async (args) => {
            const [text, conversationId, botId, needCreate, msgUuid, localConvId, useDeepThink] = args;

            // Get fp from cookie
            const fp = document.cookie.split(';')
                .map(c => c.trim())
                .find(c => c.startsWith('s_v_web_id='))
                ?.split('=')[1] || '';

            // Get device params from localStorage
            let deviceId = '';
            let webId = '';
            try {
                const samWeb = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');
                deviceId = samWeb.web_id || '';
            } catch(e) {}
            try {
                const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
                webId = tea.web_id || '';
            } catch(e) {}

            // Build query string matching real frontend
            const params = new URLSearchParams({
                aid: '497858',
                device_id: deviceId,
                device_platform: 'web',
                fp: fp,
                language: 'zh',
                pc_version: '3.19.4',
                pkg_type: 'release_version',
                real_aid: '497858',
                region: '',
                samantha_web: '1',
                sys_region: '',
                tea_uuid: webId,
                'use-olympus-account': '1',
                version_code: '20800',
                web_id: webId,
                web_tab_id: crypto.randomUUID()
            });
            const url = '/chat/completion?' + params.toString();

            const body = {
                client_meta: {
                    local_conversation_id: needCreate ? localConvId : "",
                    conversation_id: conversationId || "",
                    bot_id: botId,
                    last_section_id: "",
                    last_message_index: null
                },
                messages: [{
                    local_message_id: msgUuid,
                    content_block: [{
                        block_type: 10000,
                        content: {
                            text_block: {text: text, icon_url: "", icon_url_dark: "", summary: ""},
                            pc_event_block: ""
                        },
                        block_id: crypto.randomUUID(),
                        parent_id: "",
                        meta_info: [],
                        append_fields: []
                    }],
                    message_status: 0
                }],
                option: {
                    send_message_scene: "",
                    create_time_ms: Date.now(),
                    collect_id: "",
                    is_audio: false,
                    answer_with_suggest: false,
                    tts_switch: false,
                    need_deep_think: useDeepThink ? 1 : 0,
                    click_clear_context: false,
                    from_suggest: false,
                    is_regen: false,
                    is_replace: false,
                    disable_sse_cache: false,
                    select_text_action: "",
                    resend_for_regen: false,
                    scene_type: 0,
                    unique_key: crypto.randomUUID(),
                    start_seq: 0,
                    need_create_conversation: needCreate,
                    regen_query_id: [],
                    edit_query_id: [],
                    regen_instruction: "",
                    no_replace_for_regen: false,
                    message_from: 0,
                    shared_app_name: "",
                    shared_app_id: "",
                    sse_recv_event_options: {support_chunk_delta: true},
                    is_ai_playground: false,
                    recovery_option: {
                        is_recovery: false,
                        req_create_time_sec: Math.floor(Date.now() / 1000),
                        append_sse_event_scene: 0
                    },
                    message_storage_type: 0
                },
                ext: {
                    use_deep_think: useDeepThink ? "1" : "0",
                    fp: fp,
                    collection_id: "",
                    commerce_credit_config_enable: "0",
                    sub_conv_firstmet_type: needCreate ? "1" : "0"
                }
            };

            // Add conversation_init_option for new conversations
            if (needCreate) {
                body.option.conversation_init_option = {need_ack_conversation: true};
                body.ext.conversation_init_option = JSON.stringify({need_ack_conversation: true});
            }

            const resp = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Agw-Js-Conv': 'str, str'
                },
                body: JSON.stringify(body)
            });

            const text_resp = await resp.text();
            return {status: resp.status, body: text_resp};
        }
        """

        result = await self._page.evaluate(
            js_code,
            [text, conversation_id or "", effective_bot_id, need_create, msg_uuid, local_conv_id, use_deep_think],
        )

        if result["status"] != 200:
            yield {"error": True, "status": result["status"], "body": result["body"]}
            return

        # Parse SSE lines - track event type per event block
        current_event = ""
        for line in result["body"].split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                current_event = line[7:]
                continue
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                data["_event"] = current_event
                yield data
            except json.JSONDecodeError:
                continue

    async def send_message(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        use_deep_think: bool = False,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Send a message via UI simulation (type + Enter).

        This is the primary method - it uses the frontend's own code path,
        which handles captcha/rate-limiting automatically.

        Returns dict with keys: text, conversation_id, events
        """
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        # Navigate to correct conversation if needed
        current_url = self._page.url
        if conversation_id and conversation_id not in current_url:
            target_url = f"{DOUBAO_URL}/chat/{conversation_id}"
            await self._page.goto(target_url, wait_until="domcontentloaded")
            await self._page.wait_for_selector("textarea", timeout=10000)
            await asyncio.sleep(1)
        elif not conversation_id and "/chat/" in current_url and len(current_url) > len(CHAT_URL) + 5:
            # Navigate to fresh chat page for new conversation
            await self._page.goto(CHAT_URL, wait_until="domcontentloaded")
            await self._page.wait_for_selector("textarea", timeout=10000)
            await asyncio.sleep(1)

        # Set up route interception AFTER navigation
        captured = {"body": None, "done": asyncio.Event()}

        async def _intercept(route):
            try:
                response = await route.fetch()
                body = await response.text()
                captured["body"] = body
                captured["done"].set()
                await route.fulfill(response=response, body=body)
            except Exception as e:
                log.warning("Route intercept error: %s", e)
                try:
                    await route.continue_()
                except Exception:
                    pass

        await self._page.route("**/chat/completion*", _intercept)

        try:
            # Type and send the message
            textarea = self._page.locator("textarea").first
            await textarea.click()
            await asyncio.sleep(0.2)
            await textarea.fill(text)
            await asyncio.sleep(0.2)
            await self._page.keyboard.press("Enter")

            # Wait for response
            try:
                await asyncio.wait_for(captured["done"].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(f"Timeout waiting for response ({timeout}s)")
        finally:
            await self._page.unroute("**/chat/completion*")

        # Parse SSE response
        events = self._parse_sse(captured["body"])

        # Check for errors
        for e in events:
            if e.get("error_code"):
                raise RuntimeError(
                    f"API error {e['error_code']}: {e.get('error_msg', '')}"
                )

        # Extract text and conversation_id
        full_text = ""
        result_conv_id = conversation_id
        for e in events:
            if not result_conv_id:
                cid = self.extract_conversation_id(e)
                if cid and cid != "0":
                    result_conv_id = cid
            full_text += self._extract_text(e)

        return {
            "text": full_text,
            "conversation_id": result_conv_id,
            "events": events,
        }

    @staticmethod
    def _parse_sse(body: str) -> list:
        """Parse SSE response body into list of event dicts."""
        events = []
        current_event = ""
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    data["_event"] = current_event
                    events.append(data)
                except json.JSONDecodeError:
                    pass
        return events

    async def chat(self, text: str, **kwargs) -> Dict[str, Any]:
        """Simple chat: send message, return full text response and metadata.

        Uses UI simulation (send_message) as primary method.
        Returns dict with keys: text, conversation_id
        """
        result = await self.send_message(text, **kwargs)
        return {"text": result["text"], "conversation_id": result["conversation_id"]}

    @staticmethod
    def _extract_text(event: Dict[str, Any]) -> str:
        """Extract text from a SSE event."""
        event_type = event.get("_event", "")

        # CHUNK_DELTA is the simplest - just has {"text": "..."}
        if event_type == "CHUNK_DELTA" and "text" in event:
            return event["text"]

        # Handle STREAM_CHUNK with patch_op
        if "patch_op" in event:
            for op in event["patch_op"]:
                pv = op.get("patch_value", {})
                # content_block format (normal response)
                for block in pv.get("content_block", []):
                    content = block.get("content", {})
                    tb = content.get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                # Old format: patch_object=102, content is JSON string
                if op.get("patch_object") == 102:
                    raw = pv.get("content", "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if parsed.get("text"):
                                return parsed["text"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        # Handle STREAM_MSG_NOTIFY (first chunk)
        if event_type == "STREAM_MSG_NOTIFY":
            content = event.get("content", {})
            if isinstance(content, dict):
                # content_block format
                for block in content.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                # Old format: model_content field
                if content.get("model_content"):
                    return content["model_content"]

        return ""

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        """Extract conversation_id from SSE_ACK event."""
        ack = event.get("ack_client_meta", {})
        if ack.get("conversation_id"):
            return ack["conversation_id"]
        # Also check meta in STREAM_MSG_NOTIFY
        meta = event.get("meta", {})
        if meta.get("conversation_id"):
            return meta["conversation_id"]
        return None

    async def stop(self):
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._ready = False
        log.info("Browser client stopped")
