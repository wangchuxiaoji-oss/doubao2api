"""Playwright-based Doubao client with httpx API calls.

Architecture:
- Playwright: Login (QR scan via noVNC) + page session + bdms.frontierSign signing
- httpx: Actual API requests with streaming SSE support

ByteDance's frontend exposes window.bdms.frontierSign() which generates
X-Bogus signatures. We use Playwright only to maintain a logged-in page
and call this signing function. All actual API traffic goes through httpx.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Optional, Dict, Any, List
from urllib.parse import urlencode

import httpx
from playwright.async_api import async_playwright, BrowserContext, Page
from playwright_stealth import Stealth

log = logging.getLogger(__name__)

DOUBAO_URL = "https://www.doubao.com"
CHAT_URL = f"{DOUBAO_URL}/chat/"
COMPLETION_URL = f"{DOUBAO_URL}/chat/completion"
SAMANTHA_COMPLETION_URL = f"{DOUBAO_URL}/samantha/chat/completion"
DEFAULT_BOT_ID = "7338286299411103781"


class BrowserClient:
    """Manages Playwright for signing and httpx for API calls."""

    def __init__(self, headless: bool = True, user_data_dir: Optional[str] = None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._ready = False
        self._device_id: Optional[str] = None
        self._web_id: Optional[str] = None
        self._fp: Optional[str] = None
        # msToken rotation: updated from x-ms-token response header
        self._ms_token: str = ""
        # Robustness: failure tracking
        self._consecutive_failures: int = 0
        self._last_error_code: int = 0
        self._needs_captcha: bool = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def page(self) -> Optional[Page]:
        return self._page

    @property
    def needs_captcha(self) -> bool:
        return self._needs_captcha

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error_code(self) -> int:
        return self._last_error_code

    def record_success(self):
        """Reset failure counters on successful request."""
        self._consecutive_failures = 0
        self._last_error_code = 0
        self._needs_captcha = False

    def record_failure(self, error_code: int = 0):
        """Track consecutive failures. Mark captcha-needed on 710022004."""
        self._consecutive_failures += 1
        self._last_error_code = error_code
        if error_code == 710022004:
            self._needs_captcha = True
            log.warning("Captcha required (710022004) - marking needs_captcha=True")
        if self._consecutive_failures >= 5:
            log.error("5 consecutive failures - marking not ready")
            self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        """Launch browser, navigate to Doubao, init httpx client."""
        log.info("Starting BrowserClient (headless=%s)", self.headless)
        self._playwright = await async_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",
        ]

        if self.user_data_dir:
            self._context = await self._playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=self.headless,
                args=launch_args,
                viewport={"width": 1280, "height": 720},
                locale="zh-CN",
            )
            self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        else:
            browser = await self._playwright.chromium.launch(
                headless=self.headless, args=launch_args,
            )
            self._context = await browser.new_context(
                viewport={"width": 1280, "height": 720}, locale="zh-CN",
            )
            self._page = await self._context.new_page()

        # Stealth patches
        stealth = Stealth(navigator_languages_override=("zh-CN", "zh"))
        await stealth.apply_stealth_async(self._page)

        # Navigate
        log.info("Navigating to %s", CHAT_URL)
        await self._page.goto(CHAT_URL, wait_until="load", timeout=60000)
        await asyncio.sleep(3)

        # Init httpx
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))

        await self._check_login_state()

    async def stop(self):
        """Close browser and httpx client."""
        if self._http:
            await self._http.aclose()
            self._http = None
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._context = None
        self._playwright = None
        self._page = None
        self._ready = False
        log.info("BrowserClient stopped")

    async def is_alive(self) -> bool:
        """Check if browser process is still responsive."""
        if not self._page or not self._context:
            return False
        try:
            result = await asyncio.wait_for(
                self._page.evaluate("1+1"), timeout=5
            )
            return result == 2
        except Exception as e:
            log.warning("Browser health check failed: %s", e)
            return False

    async def restart(self):
        """Stop and restart the browser client."""
        log.info("Restarting BrowserClient...")
        await self.stop()
        await asyncio.sleep(2)
        await self.start()
        log.info("BrowserClient restarted. ready=%s", self._ready)

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    async def _check_login_state(self):
        """Check if logged in by looking for login button."""
        login_btn = self._page.locator('button:has-text("登录")')
        btn_count = await login_btn.count()
        log.info("Login check: login_button_count=%d", btn_count)

        if btn_count > 0:
            log.info("Not logged in - login button visible")
            self._ready = False
            return

        self._ready = True
        await self._extract_params()
        await self._seed_ms_token()
        await self._wait_for_signing()
        log.info("Ready! device_id=%s", self._device_id)

    async def _extract_params(self):
        """Extract device_id, web_id, fp from localStorage/cookies."""
        for _ in range(5):
            params = await self._page.evaluate("""() => {
                const result = {};
                try {
                    const samWeb = JSON.parse(localStorage.getItem('samantha_web_web_id') || '{}');
                    result.device_id = samWeb.web_id || '';
                } catch(e) {}
                try {
                    const tea = JSON.parse(localStorage.getItem('__tea_cache_tokens_497858') || '{}');
                    result.web_id = tea.web_id || '';
                } catch(e) {}
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
        log.info("Params: device_id=%s, web_id=%s, fp=%s",
                 self._device_id, self._web_id, self._fp[:20] if self._fp else "")

    async def _wait_for_signing(self):
        """Wait for bdms.frontierSign to become available."""
        for i in range(12):  # up to 60s
            has_sign = await self._page.evaluate(
                "() => typeof window.bdms?.frontierSign === 'function'"
            )
            if has_sign:
                log.info("bdms.frontierSign available after %ds", (i + 1) * 5)
                return
            await asyncio.sleep(5)
        log.warning("bdms.frontierSign not available after 60s - signing may fail")

    async def wait_for_login(self, timeout: int = 120) -> bool:
        """Wait for user to scan QR code via noVNC."""
        await self._trigger_login_dialog()
        log.info("Waiting for QR scan login (timeout=%ds)...", timeout)
        try:
            login_btn = self._page.locator('button:has-text("登录")')
            await login_btn.wait_for(state="hidden", timeout=timeout * 1000)
            await asyncio.sleep(2)
            if await login_btn.count() == 0:
                self._ready = True
                await self._extract_params()
                await self._seed_ms_token()
                await self._wait_for_signing()
                log.info("Login successful!")
                return True
            return False
        except Exception as e:
            log.error("Login timeout: %s", e)
            return False

    async def _trigger_login_dialog(self):
        """Click login button to show QR code."""
        btn = self._page.locator('button:has-text("登录")')
        if await btn.count() > 0:
            await btn.click()
            await asyncio.sleep(2)


    async def inject_cookies_and_reload(self, cookies: Dict[str, str]) -> bool:
        """Inject cookies from QR login into browser context and reload.

        After qr_login.py obtains session cookies via pure HTTP,
        this method injects them into Playwright so that bdms.frontierSign
        becomes available.

        Returns True if login state is confirmed after reload.
        """
        if not self._context or not self._page:
            log.error("inject_cookies: browser not started")
            return False

        # Build cookie list for Playwright
        pw_cookies = []
        for name, value in cookies.items():
            pw_cookies.append({
                "name": name,
                "value": value,
                "domain": ".doubao.com",
                "path": "/",
            })

        await self._context.add_cookies(pw_cookies)
        log.info("Injected %d cookies into browser context", len(pw_cookies))

        # Reload page to pick up new session
        await self._page.reload(wait_until="load", timeout=30000)
        await asyncio.sleep(3)

        # Re-check login state
        await self._check_login_state()
        return self._ready
    # ------------------------------------------------------------------
    # Signing & Cookies
    # ------------------------------------------------------------------

    async def _get_cookies_string(self) -> str:
        """Get full cookie string including httpOnly cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        return "; ".join(f"{c['name']}={c['value']}" for c in cookies)

    async def _get_csrf_token(self) -> str:
        """Get passport_csrf_token from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "passport_csrf_token":
                return c["value"]
            if c["name"] == "passport_csrf_token_default":
                return c["value"]
        return ""

    async def _seed_ms_token(self):
        """Seed initial msToken from browser cookies."""
        cookies = await self._context.cookies("https://www.doubao.com")
        for c in cookies:
            if c["name"] == "msToken":
                self._ms_token = c["value"]
                log.info("Seeded msToken from cookies (%d chars)", len(c["value"]))
                return
        log.warning("No msToken cookie found - first request may trigger rate limit")

    async def _sign_url(self, base_url: str, params: Dict[str, str]) -> str:
        """Sign a URL using bdms.frontierSign with retry on failure."""
        sorted_params = dict(sorted(params.items()))
        query_string = urlencode(sorted_params)

        last_error = None
        for attempt in range(3):
            try:
                sig = await self._page.evaluate(
                    f'window.bdms.frontierSign("{query_string}")'
                )

                x_bogus = ""
                if isinstance(sig, dict):
                    x_bogus = sig.get("X-Bogus") or sig.get("a_bogus", "")
                elif isinstance(sig, str):
                    x_bogus = sig

                if x_bogus:
                    return f"{base_url}?{query_string}&X-Bogus={x_bogus}"

                last_error = f"empty signature: {sig}"
            except Exception as e:
                last_error = str(e)
                log.warning("frontierSign attempt %d failed: %s", attempt + 1, e)

            if attempt < 2:
                await asyncio.sleep(1)

        log.error("frontierSign failed after 3 attempts: %s", last_error)
        raise RuntimeError(f"Failed to generate X-Bogus signature: {last_error}")

    def _build_query_params(self) -> Dict[str, str]:
        """Build the standard query parameters for API calls."""
        params = {
            "aid": "497858",
            "device_id": self._device_id or "",
            "device_platform": "web",
            "fp": self._fp or "",
            "language": "zh",
            "pc_version": "3.19.4",
            "pkg_type": "release_version",
            "real_aid": "497858",
            "region": "",
            "samantha_web": "1",
            "sys_region": "",
            "tea_uuid": self._web_id or "",
            "use-olympus-account": "1",
            "version_code": "20800",
            "web_id": self._web_id or "",
            "web_tab_id": str(uuid.uuid4()),
        }
        if self._ms_token:
            params["msToken"] = self._ms_token
        return params

    def _build_headers(self, cookie_str: str, csrf_token: str = "") -> Dict[str, str]:
        """Build request headers."""
        # Extract CSRF token from cookie string if not provided
        if not csrf_token:
            for part in cookie_str.split("; "):
                if part.startswith("passport_csrf_token="):
                    csrf_token = part.split("=", 1)[1]
                    break
        headers = {
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Content-Type": "application/json",
            "Cookie": cookie_str,
            "Origin": DOUBAO_URL,
            "Referer": CHAT_URL,
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "agw-js-conv": "str, str",
        }
        if csrf_token:
            headers["x-tt-passport-csrf-token"] = csrf_token
        return headers

    # ------------------------------------------------------------------
    # Chat Completion (streaming via httpx)
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a chat message and yield SSE events via httpx streaming."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        need_create = conversation_id is None or conversation_id == ""
        effective_bot_id = bot_id or DEFAULT_BOT_ID
        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id if need_create else "",
                "conversation_id": conversation_id or "",
                "bot_id": effective_bot_id,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": [{
                "local_message_id": msg_uuid,
                "content_block": [{
                    "block_type": 10000,
                    "content": {
                        "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                        "pc_event_block": "",
                    },
                    "block_id": str(uuid.uuid4()),
                    "parent_id": "",
                    "meta_info": [],
                    "append_fields": [],
                }],
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "",
                "create_time_ms": now_ms,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": False,
                "tts_switch": False,
                "need_deep_think": use_deep_think,
                "click_clear_context": False,
                "from_suggest": False,
                "is_regen": False,
                "is_replace": False,
                "disable_sse_cache": False,
                "select_text_action": "",
                "resend_for_regen": False,
                "scene_type": 0,
                "unique_key": str(uuid.uuid4()),
                "start_seq": 0,
                "need_create_conversation": need_create,
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "recovery_option": {
                    "is_recovery": False,
                    "req_create_time_sec": now_sec,
                    "append_sse_event_scene": 0,
                },
                "message_storage_type": 0,
            },
            "ext": {
                "use_deep_think": str(use_deep_think),
                "fp": self._fp or "",
                "collection_id": "",
                "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1" if need_create else "0",
            },
        }

        query_params = self._build_query_params()
        signed_url = await self._sign_url(COMPLETION_URL, query_params)
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        log.info("POST %s (conv=%s, deep_think=%s)",
                 COMPLETION_URL, conversation_id or "new", use_deep_think)

        async with self._http.stream("POST", signed_url, headers=headers,
                                     json=payload) as response:
            # Rotate msToken from response header
            new_ms_token = response.headers.get("x-ms-token", "")
            if new_ms_token:
                self._ms_token = new_ms_token
            if response.status_code != 200:
                body = await response.aread()
                error_text = body.decode(errors="ignore")
                log.error("API error %d: %s", response.status_code, error_text[:200])
                yield {"error": True, "status": response.status_code, "body": error_text}
                return

            current_event = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                    continue
                if line.startswith("id: "):
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str or data_str == "{}":
                    continue
                try:
                    data = json.loads(data_str)
                    data["_event"] = current_event
                    yield data
                except json.JSONDecodeError:
                    continue

    # ------------------------------------------------------------------
    # High-level chat helper
    # ------------------------------------------------------------------

    async def chat(
        self,
        text: str,
        conversation_id: Optional[str] = None,
        bot_id: Optional[str] = None,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Send message, collect full response. Returns {text, conversation_id}."""
        full_text = ""
        result_conv_id = conversation_id
        events = []

        async for event in self.chat_completion(
            text, conversation_id=conversation_id,
            bot_id=bot_id, use_deep_think=use_deep_think
        ):
            events.append(event)
            if event.get("error"):
                raise RuntimeError(
                    f"API error {event.get('status')}: {event.get('body', '')[:200]}"
                )
            if not result_conv_id:
                cid = self.extract_conversation_id(event)
                if cid and cid != "0":
                    result_conv_id = cid
            full_text += self._extract_text(event)

        return {"text": full_text, "conversation_id": result_conv_id}

    # ------------------------------------------------------------------
    # SSE parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(event: Dict[str, Any]) -> str:
        """Extract text content from a SSE event."""
        event_type = event.get("_event", "")

        if event_type == "CHUNK_DELTA" and "text" in event:
            return event["text"]

        if "patch_op" in event:
            for op in event["patch_op"]:
                pv = op.get("patch_value", {})
                for block in pv.get("content_block", []):
                    content = block.get("content", {})
                    tb = content.get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]
                if op.get("patch_object") == 102:
                    raw = pv.get("content", "")
                    if raw:
                        try:
                            parsed = json.loads(raw)
                            if parsed.get("text"):
                                return parsed["text"]
                        except (json.JSONDecodeError, TypeError):
                            pass

        if event_type == "STREAM_MSG_NOTIFY":
            content = event.get("content", {})
            if isinstance(content, dict):
                for block in content.get("content_block", []):
                    tb = block.get("content", {}).get("text_block", {})
                    if tb.get("text"):
                        return tb["text"]

        return ""

    @staticmethod
    def extract_conversation_id(event: Dict[str, Any]) -> Optional[str]:
        """Extract conversation_id from SSE events."""
        ack = event.get("ack_client_meta", {})
        if ack.get("conversation_id"):
            return ack["conversation_id"]
        meta = event.get("meta", {})
        if meta.get("conversation_id"):
            return meta["conversation_id"]
        return None

    # ------------------------------------------------------------------
    # Samantha endpoint (image/video/music generation)
    # ------------------------------------------------------------------

    async def _samantha_request(
        self,
        payload: Dict[str, Any],
        timeout: float = 120,
    ) -> str:
        """Send a request to /samantha/chat/completion and return raw body."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        query_params = self._build_query_params()
        signed_url = await self._sign_url(SAMANTHA_COMPLETION_URL, query_params)
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        resp = await self._http.post(
            signed_url, headers=headers, json=payload, timeout=timeout,
        )
        # Rotate msToken
        new_ms = resp.headers.get("x-ms-token", "")
        if new_ms:
            self._ms_token = new_ms
        if resp.status_code != 200:
            raise RuntimeError(
                f"samantha/chat/completion failed ({resp.status_code}): "
                f"{resp.text[:500]}"
            )

        body = resp.text
        if body.lstrip().startswith("{"):
            try:
                err = json.loads(body)
                if isinstance(err, dict) and "code" in err:
                    raise RuntimeError(
                        f"samantha auth error: code={err.get('code')} "
                        f"msg={err.get('msg') or err.get('message', '')}"
                    )
            except json.JSONDecodeError:
                pass
        return body

    @staticmethod
    def _parse_samantha_sse(raw: str) -> List[Dict[str, Any]]:
        """Parse samantha SSE body into list of event dicts."""
        events = []
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            data_str = ""
            for line in block.strip().split("\n"):
                if line.startswith("data:"):
                    data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                events.append(json.loads(data_str))
            except json.JSONDecodeError:
                continue
        return events

    async def generate_image(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate images using /samantha/chat/completion.

        Args:
            prompt: Text description of the image to generate.
            ratio: Aspect ratio ("1:1", "16:9", "9:16", "4:3", "3:4").
            ref_image_key: Optional uploaded image key for reference.

        Returns:
            Dict with 'images' list, each having url/width/height/key.
        """
        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2009,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 3,
                "skill_type_no_default": 3,
                "skill_id": "3",
                "skill_id_no_default": "3",
            },
        }

        if ref_image_key:
            message["attachments"] = [
                {"type": "image", "key": ref_image_key,
                 "extra": {"refer_types": "overall"}}
            ]

        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 3,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_image: prompt=%s, ratio=%s", prompt[:50], ratio)
        raw = await self._samantha_request(payload, timeout=120)

        # Parse response - look for content_type=2010 (image output)
        images = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_image error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") != 2010:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", []):
                if not isinstance(item, dict):
                    continue
                ori = item.get("image_ori", {}) or {}
                raw_img = item.get("image_raw", {}) or {}
                thumb = item.get("image_thumb", {}) or {}
                images.append({
                    "key": item.get("key", ""),
                    "url": ori.get("url") or raw_img.get("url") or thumb.get("url", ""),
                    "width": ori.get("width") or thumb.get("width", 0),
                    "height": ori.get("height") or thumb.get("height", 0),
                    "format": ori.get("format") or thumb.get("format", ""),
                })

        log.info("generate_image: got %d images", len(images))
        return {"images": images, "prompt": prompt}

    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate music using /samantha/chat/completion.

        Args:
            prompt: Text description of the music to generate.
            lyric: Explicit lyrics (optional).
            genre: Music genre (optional).

        Returns:
            Dict with 'tracks' list, each having audio_url/title/lyrics/duration.
        """
        import base64

        content_data: Dict[str, Any] = {"text": prompt}
        if lyric:
            content_data["lyric"] = lyric
        if genre:
            content_data["genre"] = genre

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2005,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 9,
                "skill_type_no_default": 9,
                "skill_id": "9",
                "skill_id_no_default": "9",
            },
        }

        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 9,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_music: prompt=%s", prompt[:50])
        raw = await self._samantha_request(payload, timeout=300)

        # Parse: find last content_type=2006 with video_model
        tracks = []
        final_content = None
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_music error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") not in (2006, 2004):
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            # Keep updating - we want the final (most complete) version
            final_content = content

        if not final_content:
            log.warning("generate_music: no content_type=2006 found")
            return {"tracks": [], "prompt": prompt}

        # Parse tasks
        tasks = final_content.get("tasks", {})
        if isinstance(tasks, dict):
            tasks_list = list(tasks.values())
        elif isinstance(tasks, list):
            tasks_list = tasks
        else:
            tasks_list = []

        for task in tasks_list:
            if not isinstance(task, dict):
                continue

            audio_url = ""
            duration = 0.0
            vm_str = task.get("video_model", "")
            if vm_str:
                try:
                    vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                    duration = vm.get("video_duration", 0.0)
                    vlist = vm.get("video_list", {})
                    for _q, vinfo in vlist.items():
                        main_url_b64 = vinfo.get("main_url", "")
                        if main_url_b64:
                            audio_url = base64.b64decode(main_url_b64).decode(
                                "utf-8", errors="replace"
                            )
                            break
                except (json.JSONDecodeError, Exception):
                    pass

            cover_url = ""
            cover = task.get("cover", {})
            if isinstance(cover, dict):
                cover_ori = cover.get("image_ori", {}) or {}
                cover_url = cover_ori.get("url", "")

            if audio_url or task.get("title"):
                tracks.append({
                    "audio_url": audio_url,
                    "title": task.get("title", ""),
                    "lyrics": task.get("lyric", ""),
                    "duration": duration,
                    "cover_url": cover_url,
                })

        log.info("generate_music: got %d tracks", len(tracks))
        return {"tracks": tracks, "prompt": prompt}

    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate video using /samantha/chat/completion (async 2-step).

        Args:
            prompt: Text description of the video to generate.
            ratio: Aspect ratio ("16:9", "9:16", "1:1").

        Returns:
            Dict with 'videos' list, each having video_url/cover_url/duration.
        """
        import base64

        content_data: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_data["ratio"] = ratio

        message: Dict[str, Any] = {
            "content": json.dumps(content_data, ensure_ascii=False),
            "content_type": 2020,
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 17,
                "skill_type_no_default": 17,
                "skill_id": "17",
                "skill_id_no_default": "17",
            },
        }

        payload = {
            "messages": [message],
            "completion_option": {
                "is_regen": False,
                "with_suggest": True,
                "need_create_conversation": True,
                "launch_stage": 1,
                "is_replace": False,
                "is_delete": False,
                "is_ai_playground": False,
                "memory_type": 2,
                "message_from": 0,
                "use_deep_think": False,
                "use_auto_cot": False,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
                "action_bar_skill_id": 17,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }

        log.info("generate_video: prompt=%s, ratio=%s", prompt[:50], ratio)
        raw = await self._samantha_request(payload, timeout=60)

        # Phase 1: Extract async task_id from fin_reason
        task_id = None
        text_parts = []
        for data in self._parse_samantha_sse(raw):
            et = data.get("event_type")
            if et == 2005:
                detail = data.get("event_data", "")
                raise RuntimeError(f"generate_video error: {str(detail)[:500]}")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            # Check for async task
            fin_reason = ed.get("fin_reason", {})
            if fin_reason and fin_reason.get("reason") == 1:
                async_task = fin_reason.get("async_task", {})
                task_id = async_task.get("id", "")

            # Collect text for error messages
            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue
            if msg.get("content_type") == 2001:
                content_raw = msg.get("content", "")
                if isinstance(content_raw, str):
                    try:
                        c = json.loads(content_raw)
                        text_parts.append(c.get("text", ""))
                    except json.JSONDecodeError:
                        pass

        full_text = "".join(text_parts)
        if "服务过载" in full_text or "重试" in full_text:
            raise RuntimeError("视频生成服务过载，请稍后重试")

        if not task_id:
            # Maybe sync result with content_type=2021, or just text response
            if full_text:
                return {"videos": [], "prompt": prompt, "message": full_text}
            raise RuntimeError("Video generation: no task_id returned")

        # Phase 2: Poll for result
        log.info("generate_video: polling task_id=%s", task_id)
        return await self._poll_video_result(task_id, prompt)

    async def _poll_video_result(
        self, task_id: str, prompt: str, timeout: float = 300
    ) -> Dict[str, Any]:
        """Poll /samantha/chat/completion with task_id for video result."""
        import base64

        poll_payload = {"task_id": task_id, "event_id": 0}
        query_params = self._build_query_params()
        signed_url = await self._sign_url(SAMANTHA_COMPLETION_URL, query_params)
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        resp = await self._http.post(
            signed_url, headers=headers, json=poll_payload, timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Video poll failed ({resp.status_code}): {resp.text[:500]}"
            )

        videos = []
        for data in self._parse_samantha_sse(resp.text):
            et = data.get("event_type")
            if et != 2001:
                continue

            ed = data.get("event_data", {})
            if isinstance(ed, str):
                try:
                    ed = json.loads(ed)
                except json.JSONDecodeError:
                    continue

            msg = ed.get("message", {})
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except json.JSONDecodeError:
                    continue

            if msg.get("content_type") != 2021:
                continue

            content_raw = msg.get("content", "")
            if isinstance(content_raw, str):
                try:
                    content = json.loads(content_raw)
                except json.JSONDecodeError:
                    continue
            else:
                content = content_raw

            for item in content.get("data", [content]):
                if not isinstance(item, dict):
                    continue
                video_url = item.get("video_url", "") or item.get("url", "")
                if not video_url:
                    vm_str = item.get("video_model", "")
                    if vm_str:
                        try:
                            vm = json.loads(vm_str) if isinstance(vm_str, str) else vm_str
                            vlist = vm.get("video_list", {})
                            for _q, vinfo in vlist.items():
                                main_b64 = vinfo.get("main_url", "")
                                if main_b64:
                                    video_url = base64.b64decode(main_b64).decode(
                                        "utf-8", errors="replace"
                                    )
                                    break
                        except (json.JSONDecodeError, Exception):
                            pass

                cover_url = item.get("cover_url", "") or item.get("cover", {}).get("url", "")
                if video_url:
                    videos.append({
                        "video_url": video_url,
                        "cover_url": cover_url,
                        "width": item.get("width", 0),
                        "height": item.get("height", 0),
                        "duration": item.get("duration", 0.0),
                    })

        log.info("generate_video: got %d videos", len(videos))
        return {"videos": videos, "prompt": prompt}

    # ------------------------------------------------------------------
    # File upload (TOS / ImageX flow)
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Upload a file to Doubao's storage (ByteDance TOS via ImageX proxy).

        4-step flow:
          1. POST /alice/resource/prepare_upload -> STS credentials
          2. GET  /top/v1?Action=ApplyImageUpload -> upload address
          3. POST https://{tos_host}/upload/v1/{store_uri} -> upload binary
          4. POST /top/v1?Action=CommitImageUpload -> confirm

        Returns:
            Dict with uri, name, size, file_type.
        """
        import zlib
        import hashlib
        import hmac as hmac_mod
        from datetime import datetime, timezone
        from urllib.parse import urlparse, parse_qs, quote as url_quote

        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_size = len(file_data)
        crc32 = format(zlib.crc32(file_data) & 0xFFFFFFFF, "08x")

        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/resource/prepare_upload", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        # Step 1: prepare_upload
        resp = await self._http.post(
            signed_url, headers=headers,
            json={"tenant_id": "5", "scene_id": "5", "resource_type": 1},
            timeout=30,
        )
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"prepare_upload failed: {body.get('msg', body)}")
        data = body["data"]
        service_id = data["service_id"]
        auth_token = data["upload_auth_token"]
        ak = auth_token["access_key"]
        sk = auth_token["secret_key"]
        st = auth_token["session_token"]

        # AWS V4 signing helper
        def _aws_sign_v4(method, url, req_body):
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path or "/"
            now = datetime.now(timezone.utc)
            amz_date = now.strftime("%Y%m%dT%H%M%SZ")
            date_stamp = now.strftime("%Y%m%d")
            qparams = parse_qs(parsed.query, keep_blank_values=True)
            sorted_qp = sorted((k, v[0] if v else "") for k, v in qparams.items())
            canonical_qs = "&".join(
                f"{url_quote(k, safe='~')}={url_quote(v, safe='~')}" for k, v in sorted_qp
            )
            h2s = {"host": host, "x-amz-date": amz_date}
            if st:
                h2s["x-amz-security-token"] = st
            signed_h = ";".join(sorted(h2s.keys()))
            canonical_h = "".join(f"{k}:{v}\n" for k, v in sorted(h2s.items()))
            body_b = req_body if isinstance(req_body, bytes) else req_body.encode()
            payload_hash = hashlib.sha256(body_b).hexdigest()
            cr = f"{method}\n{path}\n{canonical_qs}\n{canonical_h}\n{signed_h}\n{payload_hash}"
            scope = f"{date_stamp}/cn-north-1/imagex/aws4_request"
            cr_hash = hashlib.sha256(cr.encode()).hexdigest()
            sts = f"AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{cr_hash}"
            def _s(key, msg):
                return hmac_mod.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
            k_d = _s(f"AWS4{sk}".encode("utf-8"), date_stamp)
            k_r = _s(k_d, "cn-north-1")
            k_sv = _s(k_r, "imagex")
            k_sg = _s(k_sv, "aws4_request")
            sig = hmac_mod.new(k_sg, sts.encode("utf-8"), hashlib.sha256).hexdigest()
            auth_str = f"AWS4-HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={signed_h}, Signature={sig}"
            result = {"Authorization": auth_str, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
            if st:
                result["x-amz-security-token"] = st
            return result

        # Step 2: ApplyImageUpload
        file_ext = f".{ext}" if ext else ""
        apply_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=ApplyImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}&NeedFallback=true"
            f"&FileSize={file_size}&FileExtension={file_ext}"
            f"&s=jdnfglwfkl"
        )
        sign_h = _aws_sign_v4("GET", apply_url, "")
        sign_h["Cookie"] = cookie_str
        resp = await self._http.get(apply_url, headers=sign_h, timeout=30)
        result_data = resp.json().get("Result")
        if not result_data:
            raise RuntimeError(f"ApplyImageUpload failed: {resp.json()}")
        upload_addr = result_data["UploadAddress"]
        store_info = upload_addr["StoreInfos"][0]
        store_uri = store_info["StoreUri"]
        tos_auth = store_info["Auth"]
        session_key = upload_addr["SessionKey"]
        upload_hosts = upload_addr.get("UploadHosts", [])

        # Step 3: Upload binary to TOS
        tos_host = upload_hosts[0] if upload_hosts else "tos-mya2lf.vodupload.com"
        upload_url = f"https://{tos_host}/upload/v1/{store_uri}"
        resp = await self._http.post(
            upload_url, content=file_data,
            headers={"Authorization": tos_auth, "Content-CRC32": crc32},
            timeout=120,
        )
        tos_resp = resp.json()
        if tos_resp.get("code") != 2000:
            raise RuntimeError(f"TOS upload failed: {tos_resp}")

        # Step 4: CommitImageUpload
        commit_url = (
            f"{DOUBAO_URL}/top/v1?"
            f"Action=CommitImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}"
        )
        commit_body = json.dumps({"SessionKey": session_key})
        sign_h2 = _aws_sign_v4("POST", commit_url, commit_body)
        sign_h2["Content-Type"] = "application/json"
        sign_h2["Cookie"] = cookie_str
        resp = await self._http.post(commit_url, content=commit_body, headers=sign_h2, timeout=30)
        body = resp.json()
        results = body.get("Result", {}).get("Results", [])
        if not results or results[0].get("UriStatus") != 2000:
            raise RuntimeError(f"CommitImageUpload failed: {body}")

        log.info("File uploaded: %s -> %s", filename, store_uri)
        return {"uri": store_uri, "name": filename, "size": file_size, "file_type": ext}


    async def get_file_download_url(
        self,
        uri: str,
        expire_seconds: int = 3600,
    ) -> Dict[str, Any]:
        """Get a temporary CDN URL for a previously uploaded file."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        ext = uri.rsplit(".", 1)[-1] if "." in uri else ""
        resp = await self._http.post(
            signed_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "file",
                "format": ext,
                "expire_second": expire_seconds,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        return file_urls[0].get("main_url", "")

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Dict[str, Any]:
        """Upload an image and return metadata usable by chat/image generation."""
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        query_params = self._build_query_params()
        signed_url = await self._sign_url(
            f"{DOUBAO_URL}/samantha/pages/upload_image", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        headers.pop("Content-Type", None)
        files = {
            "data": (filename, image_bytes, f"image/{ext}"),
            "file_type": (None, ext),
        }
        resp = await self._http.post(signed_url, headers=headers, files=files, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Image upload failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"Image upload error: {body.get('msg', body)}")
        uri = body.get("data", {}).get("uri", "")
        if not uri:
            raise RuntimeError(f"Image upload returned no uri: {body}")
        query_params = self._build_query_params()
        file_url = await self._sign_url(
            f"{DOUBAO_URL}/alice/message/get_file_url", query_params
        )
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)
        resp = await self._http.post(
            file_url,
            headers=headers,
            json={
                "uris": [uri],
                "type": "image",
                "format": ext,
                "expire_second": 3600,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"get_file_url failed ({resp.status_code}): {resp.text[:500]}")
        body = resp.json()
        if body.get("code") != 0:
            raise RuntimeError(f"get_file_url error: {body.get('msg', body)}")
        file_urls = body.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise RuntimeError("get_file_url returned no file_urls")
        info = file_urls[0]
        return {
            "uri": info.get("uri", uri),
            "cdn_url": info.get("main_url", ""),
            "name": filename,
            "format": ext,
            "width": "64",
            "height": "64",
        }

    async def chat_with_file(
        self,
        text: str,
        file_uri: str,
        file_name: str,
        file_size: int,
        use_deep_think: int = 0,
    ) -> Dict[str, Any]:
        """Chat with a file attachment. The AI will read the file and answer.

        Args:
            text: Question about the file.
            file_uri: URI from upload_file().
            file_name: Original filename.
            file_size: File size in bytes.
            use_deep_think: 0=quick, 1=think, 3=expert.

        Returns:
            Dict with 'text' and 'conversation_id'.
        """
        if not self._ready:
            raise RuntimeError("Browser not ready - need login first")

        msg_uuid = str(uuid.uuid4())
        local_conv_id = f"local_{uuid.uuid4().int % 10**16}"
        now_ms = int(time.time() * 1000)
        now_sec = int(time.time())

        if isinstance(file_uri, list):
            file_refs = file_uri
        else:
            file_refs = [{"uri": file_uri, "name": file_name, "size": file_size}]
        file_attachments = []
        for file_ref in file_refs:
            file_attachments.append({
                "type": 3,
                "identifier": str(uuid.uuid4()),
                "file": {
                    "uri": file_ref.get("uri", ""),
                    "url": "",
                    "file_type": 0,
                    "name": file_ref.get("name", "file.txt"),
                    "size": int(file_ref.get("size") or 0),
                },
                "parse_state": 1,
                "review_state": 1,
                "upload_status": 1,
                "progress": 100,
                "src": "",
            })

        payload = {
            "client_meta": {
                "local_conversation_id": local_conv_id,
                "conversation_id": "",
                "bot_id": DEFAULT_BOT_ID,
                "last_section_id": "",
                "last_message_index": None,
            },
            "messages": [{
                "local_message_id": msg_uuid,
                "content_block": [
                    {
                        "block_type": 10052,
                        "content": {
                            "attachment_block": {
                                "attachments": file_attachments
                            },
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "append_fields": [],
                    },
                    {
                        "block_type": 10000,
                        "content": {
                            "text_block": {"text": text, "icon_url": "", "icon_url_dark": "", "summary": ""},
                            "pc_event_block": "",
                        },
                        "block_id": str(uuid.uuid4()),
                        "parent_id": "",
                        "meta_info": [],
                        "append_fields": [],
                    },
                ],
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "", "create_time_ms": now_ms, "collect_id": "",
                "is_audio": False, "answer_with_suggest": False, "tts_switch": False,
                "need_deep_think": use_deep_think, "click_clear_context": False,
                "from_suggest": False, "is_regen": False, "is_replace": False,
                "disable_sse_cache": False, "select_text_action": "",
                "resend_for_regen": False, "scene_type": 0,
                "unique_key": str(uuid.uuid4()), "start_seq": 0,
                "need_create_conversation": True, "regen_query_id": [],
                "edit_query_id": [], "regen_instruction": "",
                "no_replace_for_regen": False, "message_from": 0,
                "shared_app_name": "", "shared_app_id": "",
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
                "recovery_option": {"is_recovery": False, "req_create_time_sec": now_sec, "append_sse_event_scene": 0},
                "message_storage_type": 0,
            },
            "ext": {
                "use_deep_think": str(use_deep_think), "fp": self._fp or "",
                "collection_id": "", "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1",
            },
        }

        query_params = self._build_query_params()
        signed_url = await self._sign_url(COMPLETION_URL, query_params)
        cookie_str = await self._get_cookies_string()
        headers = self._build_headers(cookie_str)

        full_text = ""
        conv_id = None
        async with self._http.stream("POST", signed_url, headers=headers, json=payload) as response:
            # Rotate msToken
            new_ms = response.headers.get("x-ms-token", "")
            if new_ms:
                self._ms_token = new_ms
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(f"chat_with_file error {response.status_code}: {body.decode()[:200]}")
            current_event = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    current_event = line[7:]
                    continue
                if line.startswith("id: "):
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if not data_str or data_str == "{}":
                    continue
                try:
                    data = json.loads(data_str)
                    data["_event"] = current_event
                    full_text += self._extract_text(data)
                    if not conv_id:
                        cid = self.extract_conversation_id(data)
                        if cid and cid != "0":
                            conv_id = cid
                except json.JSONDecodeError:
                    continue

        return {"text": full_text, "conversation_id": conv_id}
