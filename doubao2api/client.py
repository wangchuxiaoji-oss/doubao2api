"""
Doubao Chat API Client - reverse-engineered desktop client protocol.

Supports three endpoint families:
  - stream_call_bot (Alice legacy)
  - /samantha/chat/completion
  - /chat/completion (unified main entry, desktop client)
"""
import base64
import hashlib
import hmac as hmac_mod
import json
import logging
import os
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Union
from urllib.parse import urlencode, quote, urlparse, parse_qs

import aiohttp

from .sse import iter_sse_events

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .captcha_handler import CaptchaHandler

RISK_ERROR_CODES = {710022002, 710022004}


class DoubaoChatError(RuntimeError):
    pass


class DoubaoRateLimitError(DoubaoChatError):
    """Raised when the server returns a risk-control / rate-limit error.

    Attributes:
        code:        The numeric error code (e.g. 710022004).
        extra:       Full ``extra`` dict from the STREAM_ERROR payload.
        verify_data: The raw ``decision`` JSON string if the challenge is of
                     type ``"verify"``; empty string otherwise.
    """

    def __init__(self, code: int, msg: str, extra: Optional[Dict[str, Any]] = None):
        self.code = code
        self.extra: Dict[str, Any] = extra or {}
        decision_str: str = self.extra.get("decision", "")
        try:
            decision: Dict[str, Any] = json.loads(decision_str) if decision_str else {}
        except Exception:
            decision = {}
        self.verify_data: str = decision_str if decision.get("type") == "verify" else ""
        super().__init__(f"Rate limited ({code}): {msg}")


@dataclass
class ChatMessage:
    """A single message in a stream_call_bot response."""
    message_id: str = ""
    reply_id: str = ""
    content_type: int = 0
    text: str = ""
    status: int = 0
    chunk_seq: int = -1
    suggestions: List[str] = field(default_factory=list)
    think: str = ""
    event_type: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_user_echo(self) -> bool:
        return self.reply_id == "0"

    @property
    def is_text_chunk(self) -> bool:
        return self.content_type == 1 and not self.is_user_echo

    @property
    def is_answer_chunk(self) -> bool:
        # In Samantha deep-think mode, final answer text can arrive as
        # content_type=2008 with content.text.
        return self.content_type in (1, 2001, 10000, 2008) and not self.is_user_echo

    @property
    def is_suggestion(self) -> bool:
        return self.content_type in (2, 2002)

    @property
    def is_think_chunk(self) -> bool:
        return self.content_type in (2008, 10040) and bool(self.think)


@dataclass
class ChatResult:
    """Aggregated result of a chat completion."""
    text: str = ""
    user_message_id: str = ""
    bot_message_id: str = ""
    suggestions: List[str] = field(default_factory=list)
    thinking_text: str = ""
    events: List[ChatMessage] = field(default_factory=list)


@dataclass
class CompletionChunk:
    """A single incremental chunk from /chat/completion SSE stream.

    The thinking state machine works as follows:
      1st BLOCK_THINKING(10040) -> enter thinking phase
      2nd BLOCK_THINKING(10040) -> exit thinking phase
    Text arriving between the two markers is thinking content;
    text arriving after the second marker is the answer.
    """
    text: str = ""
    thinking: str = ""
    block_type: int = 0
    tool_info: str = ""
    search_info: Optional[Dict[str, Any]] = None
    error_code: int = 0
    error_msg: str = ""
    is_done: bool = False
    conversation_id: str = ""


@dataclass
class CompletionResult:
    """Aggregated result of a /chat/completion call."""
    text: str = ""
    thinking_text: str = ""
    tool_events: List[str] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    conversation_id: str = ""


@dataclass
class GeneratedImage:
    """A single generated image from the image generation API.

    Note: All URLs contain watermarks applied at CDN level (ImageX tplv templates).
    This is enforced server-side and cannot be bypassed — URL signatures are bound
    to the full path including the watermark template.

    Fields:
        ori_url: "download" quality, ~tplv-*-image_dld_watermark (largest, has watermark)
        raw_url: "preview" quality, ~tplv-*-image_pre_watermark (has watermark)
        thumb_url: thumbnail with watermark (smallest)
    """
    key: str = ""
    thumb_url: str = ""
    ori_url: str = ""
    raw_url: str = ""
    width: int = 0
    height: int = 0
    format: str = ""


@dataclass
class ImageGenerationResult:
    """Result of an image generation call."""
    images: List[GeneratedImage] = field(default_factory=list)
    prompt: str = ""


@dataclass
class GeneratedVideo:
    """A single generated video from the video generation API."""
    video_url: str = ""
    cover_url: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0


@dataclass
class VideoGenerationResult:
    """Result of a video generation call."""
    videos: List[GeneratedVideo] = field(default_factory=list)
    prompt: str = ""


@dataclass
class GeneratedMusic:
    """A single generated music track from the music generation API.

    The audio URL is extracted from the video_model field (base64-encoded in response).
    """
    audio_url: str = ""
    title: str = ""
    duration: float = 0.0
    lyrics: str = ""
    cover_url: str = ""
    vid: str = ""


@dataclass
class MusicGenerationResult:
    """Result of a music generation call."""
    tracks: List[GeneratedMusic] = field(default_factory=list)
    prompt: str = ""


@dataclass
class UploadedFile:
    """Metadata for a file uploaded via upload_file().

    Use this object with chat_stream_completion / completion to attach
    files to a chat message.
    """
    uri: str = ""
    name: str = ""
    size: int = 0
    file_type: str = ""


DEFAULT_BOT_ID = "7234781073513644036"
EXTENSION_BOT_ID = "7338286299411103781"

# Supported file extensions for upload (from frontend source)
UPLOAD_SUPPORTED_EXTENSIONS = {
    "pdf", "txt", "csv", "docx", "doc", "xlsx", "xls", "pptx", "ppt",
    "md", "mobi", "epub", "py", "java", "js", "ts", "c", "cpp", "h",
    "hpp", "html", "css", "php", "rb", "pl", "sh", "bash", "swift",
    "kt", "go", "dart", "scala", "cs", "xaml", "vue", "json", "yaml",
    "yml", "xml", "env", "ini", "toml", "plist", "feature", "bat",
    "cmd", "ps1", "vbs", "lua", "mod", "sum", "png", "jpeg", "jpg", "webp",
    "proto", "dockerfile", "rs", "tsx", "jsx",
}


def _aws_sign_v4(
    method: str,
    url: str,
    body: Union[bytes, str],
    access_key: str,
    secret_key: str,
    session_token: str,
    region: str = "cn-north-1",
    service: str = "imagex",
) -> Dict[str, str]:
    """Generate AWS Signature V4 headers for ByteDance ImageX proxy."""
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or "/"
    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    # Canonical query string (sorted)
    query_params = parse_qs(parsed.query, keep_blank_values=True)
    sorted_params = sorted((k, v[0] if v else "") for k, v in query_params.items())
    canonical_qs = "&".join(
        f"{quote(k, safe='~')}={quote(v, safe='~')}" for k, v in sorted_params
    )

    # Headers to sign
    headers_to_sign: Dict[str, str] = {"host": host, "x-amz-date": amz_date}
    if session_token:
        headers_to_sign["x-amz-security-token"] = session_token
    signed_headers = ";".join(sorted(headers_to_sign.keys()))
    canonical_headers = "".join(
        f"{k}:{v}\n" for k, v in sorted(headers_to_sign.items())
    )

    # Payload hash
    body_bytes = body if isinstance(body, bytes) else body.encode()
    payload_hash = hashlib.sha256(body_bytes).hexdigest()

    # Canonical request
    canonical_request = (
        f"{method}\n{path}\n{canonical_qs}\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    cr_hash = hashlib.sha256(canonical_request.encode()).hexdigest()
    string_to_sign = f"AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{cr_hash}"

    # Derive signing key
    def _sign(key: bytes, msg: str) -> bytes:
        return hmac_mod.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k_date = _sign(f"AWS4{secret_key}".encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region)
    k_service = _sign(k_region, service)
    k_signing = _sign(k_service, "aws4_request")

    signature = hmac_mod.new(
        k_signing, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    auth = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    result = {
        "Authorization": auth,
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }
    if session_token:
        result["x-amz-security-token"] = session_token
    return result


class DoubaoChatClient:
    """
    Async client for Doubao's internal chat API (stream_call_bot).

    Usage:
        async with DoubaoChatClient(cookies, ms_token=token) as client:
            result = await client.chat("你好")
            print(result.text)

            # Or stream:
            async for chunk in client.chat_stream("讲个故事"):
                print(chunk.text, end="", flush=True)
    """

    BASE_URL = "https://www.doubao.com"

    def __init__(
        self,
        cookies: Dict[str, str],
        ms_token: str,
        device_id: str = "714003710229497",
        web_id: str = "7604137868021548590",
        fp: str = "verify_mlcfw5f7_TPq0YmFD_NrsC_4RuQ_BJPg_M5W7i58I7wV0",
        bot_id: str = DEFAULT_BOT_ID,
        timeout_seconds: int = 120,
        captcha_handler: Union[None, str, "CaptchaHandler"] = "auto",
        max_captcha_retries: int = 3,
    ) -> None:
        """
        Args:
            captcha_handler:
                ``"auto"`` (default) — lazily create an ``AutoCaptchaHandler``
                that opens a browser tab when a verify challenge is received.
                ``None`` — disable automatic handling; ``DoubaoRateLimitError``
                is raised immediately (preserving the old behaviour).
                A custom ``CaptchaHandler`` instance — used as-is.
            max_captcha_retries: Maximum number of automatic captcha retries
                before giving up and raising ``DoubaoRateLimitError``.
        """
        if not cookies:
            raise ValueError("cookies are required")

        self.cookies = cookies
        self.ms_token = ms_token
        self.device_id = device_id
        self.web_id = web_id
        self.fp = fp
        self.bot_id = bot_id
        self.max_captcha_retries = max_captcha_retries
        # Use sock_read for SSE streams (reset on each chunk/heartbeat).
        # total=None avoids cutting off long expert-mode responses that
        # can exceed 3-4 minutes with multiple search rounds.
        self.timeout = aiohttp.ClientTimeout(
            total=None, sock_read=timeout_seconds
        )
        self._session: Optional[aiohttp.ClientSession] = None

        # Resolve captcha handler
        if captcha_handler == "auto":
            from .captcha_handler import AutoCaptchaHandler
            self._captcha_handler: Optional["CaptchaHandler"] = AutoCaptchaHandler()
        elif captcha_handler is None:
            self._captcha_handler = None
        else:
            self._captcha_handler = captcha_handler  # type: ignore[assignment]

    async def __aenter__(self) -> "DoubaoChatClient":
        self._session = aiohttp.ClientSession(
            cookies=self.cookies,
            timeout=self.timeout,
            read_bufsize=2**20,  # 1MB; default 64KB causes "Chunk too big" for long SSE lines
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/135.0.0.0 Safari/537.36"
                ),
                "Content-Type": "application/json",
                "Origin": "https://www.doubao.com",
                "Referer": "https://www.doubao.com/chat",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session:
            await self._session.close()
        self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError(
                "Client not initialized. Use: async with DoubaoChatClient(...) as client:"
            )
        return self._session

    @staticmethod
    def _check_gateway_error(raw: str) -> None:
        """Check raw SSE text for gateway-error event and raise if found."""
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            event_name = ""
            data_str = ""
            for line in block.strip().split("\n"):
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
            if event_name == "gateway-error" and data_str:
                try:
                    err_obj = json.loads(data_str)
                    code = err_obj.get("code", "")
                    msg = err_obj.get("message", data_str)
                except json.JSONDecodeError:
                    code, msg = "", data_str
                raise DoubaoChatError(
                    f"gateway-error: code={code} message={msg}"
                )

    def _security_params(self) -> Dict[str, str]:
        params = {
            "aid": "582478",
            "real_aid": "582478",
            "device_id": self.device_id,
            "tea_uuid": self.device_id,
            "web_id": self.web_id,
            "device_platform": "web",
            "language": "zh",
            "region": "CN",
            "sys_region": "CN",
            "pkg_type": "release_version",
            "version_code": "20800",
            "pc_version": "2.1.7",
            "chromium_version": "135.0.7049.72",
            "client_platform": "pc_client",
            "runtime": "web",
            "runtime_version": "3.5.4",
            "samantha_web": "1",
            "use-olympus-account": "1",
            "fp": self.fp,
            "web_tab_id": str(uuid.uuid4()),
        }
        # Only include msToken if we have a real one;
        # empty/fake values trigger rate limiting (710022002)
        if self.ms_token:
            params["msToken"] = self.ms_token
        return params

    @staticmethod
    def _encode_payload(obj: Dict[str, Any]) -> str:
        return base64.b64encode(
            json.dumps(obj, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")

    def _build_inner_payload(
        self,
        text: str,
        bot_id: Optional[str] = None,
        conversation_id: str = "0",
        section_id: str = "0",
    ) -> Dict[str, Any]:
        return {
            "event_type": 1,
            "message": {
                "conversation_id": conversation_id,
                "section_id": section_id,
                "local_message_id": str(uuid.uuid4()),
                "content_type": 1,
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "reply_id": "",
                "ext": {
                    "origin": "https://www.doubao.com",
                    "stream": "1",
                    "answer_with_suggest": "1",
                    "browser_language": "zh-CN",
                },
                "local_conversation_id": "0",
                "bot_id": bot_id or self.bot_id,
                "meta_infos": [],
            },
        }

    @staticmethod
    def _parse_alice_sse_message(data: Dict[str, Any]) -> ChatMessage:
        msg = data.get("message", {})
        content_type = msg.get("content_type", 0)
        text = ""
        suggestions: List[str] = []

        try:
            content = json.loads(msg.get("content", "{}"))
            if content_type == 1:
                text = content.get("text", "")
            elif content_type == 2:
                suggestions = content.get("suggest", [])
        except (json.JSONDecodeError, TypeError):
            pass

        return ChatMessage(
            message_id=msg.get("message_id", ""),
            reply_id=msg.get("reply_id", "0"),
            content_type=content_type,
            text=text,
            status=msg.get("status", 0),
            chunk_seq=msg.get("chunk_seq", -1),
            suggestions=suggestions,
            event_type=data.get("event_type", 0),
            raw=data,
        )

    @staticmethod
    def _parse_samantha_sse_message(data: Dict[str, Any]) -> Optional[ChatMessage]:
        event_type = data.get("event_type", 0)
        event_data = data.get("event_data", {})
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except json.JSONDecodeError:
                event_data = {}

        if not isinstance(event_data, dict):
            return None

        # CMPL (2001) carries message chunks, FIN (2003) marks end.
        msg = event_data.get("message", {})
        if isinstance(msg, str):
            try:
                msg = json.loads(msg)
            except json.JSONDecodeError:
                msg = {}

        if not isinstance(msg, dict):
            return None

        content_type = msg.get("content_type", 0)
        content_raw = msg.get("content", {})
        if isinstance(content_raw, str):
            try:
                content = json.loads(content_raw)
            except json.JSONDecodeError:
                content = {}
        elif isinstance(content_raw, dict):
            content = content_raw
        else:
            content = {}

        text = ""
        think = ""
        suggestions: List[str] = []

        if content_type in (2001, 2003, 10000):
            text = content.get("text", "")
        elif content_type == 2008:
            # Deep-think chunks can carry either think text (content.think)
            # or final answer text (content.text).
            think = content.get("think", "")
            text = content.get("text", "")
        elif content_type == 10040:
            think = content.get("think", "")
            if not think:
                think = (
                    content.get("finish_title", "")
                    or content.get("title", "")
                    or content.get("text", "")
                )
        elif content_type == 2002:
            suggest_value = content.get("suggest", [])
            if isinstance(suggest_value, list):
                suggestions = suggest_value
            elif isinstance(suggest_value, str):
                suggestions = [suggest_value]

        return ChatMessage(
            message_id=msg.get("message_id", ""),
            reply_id=msg.get("reply_id", ""),
            content_type=content_type,
            text=text,
            think=think,
            status=event_data.get("status", 0),
            chunk_seq=msg.get("chunk_seq", -1),
            suggestions=suggestions,
            event_type=event_type,
            raw=data,
        )

    async def chat_stream(
        self,
        text: str,
        bot_id: Optional[str] = None,
        conversation_id: str = "0",
        section_id: str = "0",
    ) -> AsyncIterator[ChatMessage]:
        """
        Send a message and yield ChatMessage objects as they arrive.

        Yields user echo first, then bot reply chunks, then suggestions.
        """
        inner = self._build_inner_payload(text, bot_id, conversation_id, section_id)
        encoded = self._encode_payload(inner)
        params = self._security_params()
        url = f"{self.BASE_URL}/alice/message/stream_call_bot?{urlencode(params)}"

        async with self.session.post(
            url,
            json={"payload": encoded},
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"stream_call_bot failed ({resp.status}): {error_text[:500]}"
                )

            async for event in iter_sse_events(resp.content):
                if event["event"] == "done":
                    return
                if event["event"] == "json" and event.get("json"):
                    yield self._parse_alice_sse_message(event["json"])

    async def chat_stream_samantha(
        self,
        text: str,
        bot_id: Optional[str] = None,
        use_deep_think: bool = False,
        use_auto_cot: bool = False,
    ) -> AsyncIterator[ChatMessage]:
        """
        Stream from /samantha/chat/completion.

        Think traces are in CMPL payloads with content_type=2008 and content.think.
        """
        url = (
            f"{self.BASE_URL}/samantha/chat/completion?"
            f"{urlencode(self._security_params())}"
        )

        sent_event = {
            "messages": [
                {
                    "content": json.dumps({"text": text}, ensure_ascii=False),
                    "content_type": 2001,
                    "attachments": [],
                    "references": [],
                }
            ],
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
                "use_deep_think": use_deep_think,
                "use_auto_cot": use_auto_cot,
                "resend_for_regen": False,
                "enable_commerce_credit": False,
            },
            "evaluate_option": {"web_ab_params": ""},
            "local_conversation_id": str(uuid.uuid4()),
            "local_message_id": str(uuid.uuid4()),
        }
        if bot_id:
            sent_event["bot_id"] = bot_id

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Agw-Js-Conv": "str",
        }

        async with self.session.post(
            url,
            data=json.dumps(sent_event, ensure_ascii=False),
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"chat/completion failed ({resp.status}): {error_text[:500]}"
                )

            raw_text = (await resp.read()).decode("utf-8", errors="replace")

            # Check if response is a JSON error (not SSE) — e.g. session expired
            if raw_text.lstrip().startswith("{"):
                try:
                    err_obj = json.loads(raw_text)
                    if isinstance(err_obj, dict) and "code" in err_obj:
                        code = err_obj.get("code", 0)
                        msg = err_obj.get("msg") or err_obj.get("message", raw_text[:200])
                        raise DoubaoChatError(
                            f"samantha/chat/completion auth error: code={code} msg={msg}"
                        )
                except json.JSONDecodeError:
                    pass

            for block in raw_text.split("\n\n"):
                if not block.strip():
                    continue

                event_name = "message"
                data_str = ""
                for line in block.strip().split("\n"):
                    if line.startswith("event:"):
                        event_name = line[6:].strip() or "message"
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()

                if not data_str:
                    continue

                # Gateway-level auth error (session expired / user invalid)
                if event_name == "gateway-error":
                    try:
                        err_obj = json.loads(data_str)
                        code = err_obj.get("code", "")
                        msg = err_obj.get("message", data_str)
                    except json.JSONDecodeError:
                        code, msg = "", data_str
                    raise DoubaoChatError(
                        f"gateway-error: code={code} message={msg}"
                    )

                if event_name not in ("message", "json"):
                    continue

                try:
                    data_json = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                parsed = self._parse_samantha_sse_message(data_json)
                if parsed is None:
                    event_json = data_json
                    if event_json.get("event_type") == 2005:
                        detail = event_json.get("event_data", "")
                        raise DoubaoChatError(
                            f"chat/completion returned ERR event: {str(detail)[:500]}"
                        )
                    continue
                if parsed.event_type == 2003:
                    return
                yield parsed

    async def chat(
        self,
        text: str,
        bot_id: Optional[str] = None,
        conversation_id: str = "0",
        section_id: str = "0",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        """
        Send a message and collect the full response.

        Args:
            text: The message text to send.
            bot_id: Override the default bot_id.
            conversation_id: Conversation ID ("0" for stateless).
            section_id: Section ID ("0" for default).
            on_chunk: Optional callback called with each text chunk as it arrives.

        Returns:
            ChatResult with aggregated text, message IDs, and suggestions.
        """
        result = ChatResult()
        text_parts: List[str] = []

        async for msg in self.chat_stream(text, bot_id, conversation_id, section_id):
            result.events.append(msg)

            if msg.is_user_echo and not result.user_message_id:
                result.user_message_id = msg.message_id
                continue

            if msg.is_text_chunk:
                if not result.bot_message_id:
                    result.bot_message_id = msg.message_id
                text_parts.append(msg.text)
                if on_chunk and msg.text:
                    on_chunk(msg.text)

            if msg.is_suggestion and msg.suggestions:
                result.suggestions.extend(msg.suggestions)

        result.text = "".join(text_parts)
        return result

    async def chat_samantha(
        self,
        text: str,
        bot_id: Optional[str] = None,
        use_deep_think: bool = False,
        use_auto_cot: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_think_chunk: Optional[Callable[[str], None]] = None,
    ) -> ChatResult:
        """
        Send a message via /samantha/chat/completion and collect full response.
        """
        result = ChatResult()
        text_parts: List[str] = []
        think_parts: List[str] = []

        async for msg in self.chat_stream_samantha(
            text=text,
            bot_id=bot_id,
            use_deep_think=use_deep_think,
            use_auto_cot=use_auto_cot,
        ):
            result.events.append(msg)

            if msg.is_answer_chunk and msg.text:
                if not result.bot_message_id:
                    result.bot_message_id = msg.message_id
                text_parts.append(msg.text)
                if on_chunk:
                    on_chunk(msg.text)

            if msg.is_think_chunk and msg.think:
                think_parts.append(msg.think)
                if on_think_chunk:
                    on_think_chunk(msg.think)

            if msg.is_suggestion and msg.suggestions:
                result.suggestions.extend(msg.suggestions)

        result.text = "".join(text_parts)
        result.thinking_text = "".join(think_parts)
        return result

    # ── /chat/completion (unified main entry) ──────────────────

    @staticmethod
    def _build_completion_content_blocks(
        text: str,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> List[Dict[str, Any]]:
        """Build content_block list for /chat/completion payload.

        Args:
            text: User message text.
            image_attachments: Optional list of dicts with keys
                ``uri``, ``cdn_url``, ``width``, ``height``, ``format``, ``name``.
            file_attachments: Optional list of UploadedFile objects from upload_file().
        """
        blocks: List[Dict[str, Any]] = []

        # File attachments (type=3) — must come before text block
        if file_attachments:
            attachments_list = []
            for f in file_attachments:
                attachments_list.append({
                    "type": 3,
                    "identifier": str(uuid.uuid4()),
                    "file": {
                        "uri": f.uri,
                        "url": "",
                        "file_type": 0,
                        "name": f.name,
                        "size": f.size,
                    },
                    "parse_state": 1,
                    "review_state": 1,
                    "upload_status": 1,
                    "progress": 100,
                    "src": "",
                })
            blocks.append({
                "block_type": 10052,
                "content": {
                    "attachment_block": {"attachments": attachments_list},
                    "pc_event_block": "",
                },
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
            })

        # Image attachments (type=1)
        for img in image_attachments or []:
            blocks.append({
                "block_type": 10052,
                "content": {
                    "attachment_block": {
                        "attachments": [{
                            "type": 1,
                            "identifier": str(uuid.uuid4()),
                            "image": {
                                "name": img.get("name", "image.png"),
                                "uri": img["uri"],
                                "image_ori": {
                                    "url": img["cdn_url"],
                                    "width": int(img.get("width", 64)),
                                    "height": int(img.get("height", 64)),
                                    "format": img.get("format", "png"),
                                    "url_formats": {},
                                },
                            },
                            "parse_state": 0,
                            "review_state": 0,
                        }],
                    },
                    "pc_event_block": "",
                },
                "block_id": str(uuid.uuid4()),
                "parent_id": "",
                "meta_info": [],
                "append_fields": [],
                "is_finish": True,
                "patch_type": 2,
            })
        blocks.append({
            "block_type": 10000,
            "content": {
                "text_block": {
                    "text": text,
                    "icon_url": "",
                    "icon_url_dark": "",
                    "summary": "",
                },
                "pc_event_block": "",
            },
            "block_id": str(uuid.uuid4()),
            "parent_id": "",
            "meta_info": [],
            "append_fields": [],
            "is_finish": True,
            "patch_type": 2,
        })
        return blocks

    def _build_completion_payload(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> Dict[str, Any]:
        content_blocks = self._build_completion_content_blocks(
            text, image_attachments, file_attachments,
        )
        return {
            "client_meta": {
                "local_conversation_id": f"local_{uuid.uuid4().hex[:16]}",
                "conversation_id": "",
                "bot_id": bot_id or EXTENSION_BOT_ID,
                "last_section_id": "",
                "last_message_index": 0,
            },
            "messages": [{
                "local_message_id": str(uuid.uuid4()),
                "content_block": content_blocks,
                "message_status": 0,
            }],
            "option": {
                "send_message_scene": "",
                "create_time_ms": 0,
                "collect_id": "",
                "is_audio": False,
                "answer_with_suggest": True,
                "tts_switch": False,
                "need_deep_think": need_deep_think,
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
                "need_create_conversation": True,
                "conversation_init_option": {"need_ack_conversation": True},
                "regen_query_id": [],
                "edit_query_id": [],
                "regen_instruction": "",
                "no_replace_for_regen": False,
                "message_from": 0,
                "shared_app_name": "",
                "action_bar_skill_id": 0,
                "sse_recv_event_options": {"support_chunk_delta": True},
                "is_ai_playground": False,
            },
            "chat_ability": {},
            "ext": {
                "use_deep_think": str(need_deep_think),
                "fp": self.fp,
                "use_submit_pipeline": "1",
                "commerce_credit_config_enable": "0",
                "sub_conv_firstmet_type": "1",
            },
        }

    async def _chat_stream_completion_once(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Single-attempt stream from /chat/completion. May raise DoubaoRateLimitError."""
        payload = self._build_completion_payload(
            text, need_deep_think, bot_id, image_attachments, file_attachments,
        )
        params = self._security_params()
        url = f"{self.BASE_URL}/chat/completion?{urlencode(params)}"

        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-tt-passport-csrf-token": (
                self.cookies.get("passport_csrf_token")
                or self.cookies.get("passport_csrf_token_default")
                or ""
            ),
        }

        thinking_count = 0
        in_thinking = False

        async with self.session.post(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"/chat/completion failed ({resp.status}): {error_text[:500]}"
                )

            # Check if response is a JSON error instead of SSE stream
            # (e.g. session expired: {"code":710012001,"msg":"登录已过期"})
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                body_text = await resp.text()
                try:
                    err_obj = json.loads(body_text)
                    if isinstance(err_obj, dict) and "code" in err_obj:
                        code = err_obj.get("code", 0)
                        msg = err_obj.get("msg") or err_obj.get("message", body_text[:200])
                        raise DoubaoChatError(
                            f"/chat/completion auth error: code={code} msg={msg}"
                        )
                except json.JSONDecodeError:
                    pass
                # Check for gateway-error in case body is SSE-like
                self._check_gateway_error(body_text)
                # Unknown non-SSE response — raise generic error
                raise DoubaoChatError(
                    f"/chat/completion unexpected response (content-type={content_type}): "
                    f"{body_text[:300]}"
                )

            async for event in iter_sse_events(resp.content):
                obj = event.get("json")
                if obj is None:
                    continue

                sse_event_name = event.get("event", "")

                # Gateway-level auth error (session invalid / user invalid)
                if sse_event_name == "gateway-error":
                    code = obj.get("code", "")
                    msg = obj.get("message", str(obj))
                    raise DoubaoChatError(
                        f"gateway-error: code={code} message={msg}"
                    )

                # Capture conversation_id from SSE_ACK event
                if sse_event_name == "SSE_ACK":
                    ack = obj.get("ack_client_meta", {})
                    if isinstance(ack, dict):
                        cid = str(ack.get("conversation_id", "")).strip()
                        if cid and cid != "0":
                            yield CompletionChunk(conversation_id=cid)
                    continue

                if sse_event_name == "STREAM_ERROR" or "error_code" in obj:
                    code = int(obj.get("error_code", 0))
                    msg = str(obj.get("error_msg", ""))
                    if code in RISK_ERROR_CODES:
                        raise DoubaoRateLimitError(code, msg, extra=obj.get("extra"))
                    if code:
                        yield CompletionChunk(error_code=code, error_msg=msg)
                        continue

                # --- chunk_delta: compact {"text": "..."} (exclude error payloads) ---
                if (
                    "text" in obj
                    and isinstance(obj.get("text"), str)
                    and "error_code" not in obj
                ):
                    t = obj["text"]
                    if in_thinking:
                        yield CompletionChunk(thinking=t, block_type=10040)
                    else:
                        yield CompletionChunk(text=t, block_type=10000)
                    continue

                # --- process content_block arrays ---
                def _iter_blocks(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
                    for patch in data.get("patch_op", []):
                        pv = patch.get("patch_value", {})
                        yield from pv.get("content_block", [])
                    dc = data.get("content", {})
                    if isinstance(dc, dict):
                        yield from dc.get("content_block", [])

                for cb in _iter_blocks(obj):
                    bt = cb.get("block_type", 0)
                    content = cb.get("content", {})

                    if bt == 10040:
                        thinking_count += 1
                        in_thinking = thinking_count == 1
                        yield CompletionChunk(block_type=bt)

                    elif bt == 10000:
                        tb = content.get("text_block", {})
                        if isinstance(tb, dict) and tb.get("text"):
                            t = tb["text"]
                            if in_thinking:
                                yield CompletionChunk(
                                    thinking=t, block_type=bt,
                                )
                            else:
                                yield CompletionChunk(text=t, block_type=bt)

                    elif bt == 10101:
                        lb = content.get("loading_block", {})
                        tl = lb.get("text_loading", {})
                        if tl.get("text"):
                            yield CompletionChunk(
                                tool_info=tl["text"], block_type=bt,
                            )

                    elif bt == 10024:
                        gtb = content.get("generic_tool_block", {})
                        title = gtb.get("title", "")
                        if title:
                            yield CompletionChunk(
                                tool_info=f"[tool] {title}",
                                block_type=bt,
                            )

                    elif bt == 10025:
                        sqrb = content.get("search_query_result_block", {})
                        if sqrb:
                            queries = sqrb.get("queries", [])
                            summary = sqrb.get("summary", "")
                            results = sqrb.get("results", [])
                            is_finish = cb.get("is_finish", False)
                            block_id = cb.get("block_id", "")
                            parsed_results = [
                                {
                                    "title": r.get("text_card", {}).get("title", ""),
                                    "url": r.get("text_card", {}).get("url", ""),
                                    "summary": r.get("text_card", {}).get("summary", ""),
                                    "source": r.get("text_card", {}).get("source_name", ""),
                                }
                                for r in results
                                if r.get("text_card")
                            ]
                            search_data = {
                                "type": "search",
                                "summary": summary,
                                "queries": queries,
                                "is_finish": is_finish,
                                "block_id": block_id,
                                "results": parsed_results,
                            }
                            # Always emit search_info when there are results,
                            # so the consumer can track incremental updates.
                            # The consumer (_stream_chat) deduplicates by block_id.
                            if parsed_results:
                                yield CompletionChunk(
                                    tool_info=f"[search] {summary}",
                                    search_info=search_data,
                                    block_type=bt,
                                )
                            else:
                                # Still emit tool_info for loading indicators
                                yield CompletionChunk(
                                    tool_info=f"[search] {summary}",
                                    block_type=bt,
                                )

                # --- process patch_op content strings (new STREAM_CHUNK format) ---
                for patch in obj.get("patch_op", []):
                    pv = patch.get("patch_value", {})
                    if isinstance(pv, dict) and "content" in pv:
                        content_str = pv.get("content", "")
                        if isinstance(content_str, str) and content_str:
                            try:
                                content_obj = json.loads(content_str)
                                t = content_obj.get("text", "")
                                if t:
                                    if in_thinking:
                                        yield CompletionChunk(thinking=t, block_type=10000)
                                    else:
                                        yield CompletionChunk(text=t, block_type=10000)
                            except (json.JSONDecodeError, TypeError):
                                pass

                if obj.get("verbose_type"):
                    yield CompletionChunk(
                        tool_info=(
                            f"[{obj['verbose_type']}] "
                            f"{str(obj.get('verbose_payload', ''))[:200]}"
                        ),
                    )

        yield CompletionChunk(is_done=True)

    async def chat_stream_completion(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List["UploadedFile"]] = None,
    ) -> AsyncIterator[CompletionChunk]:
        """Stream from /chat/completion with automatic captcha handling.

        Yields ``CompletionChunk`` objects.  When a rate-limit / verify challenge
        is returned by the server, the configured ``captcha_handler`` is invoked
        and the request is retried transparently (up to ``max_captcha_retries``
        times).  If no handler is set or retries are exhausted, a
        ``DoubaoRateLimitError`` is raised.

        Args:
            text: User message text.
            need_deep_think: 0=quick, 1=think, 2=auto, 3=expert.
            bot_id: Override default bot_id.
            image_attachments: Pre-uploaded images (see ``upload_image``).
            file_attachments: Pre-uploaded files (see ``upload_file``).
        """
        for attempt in range(self.max_captcha_retries + 1):
            has_yielded_content = False
            try:
                async for chunk in self._chat_stream_completion_once(
                    text, need_deep_think, bot_id, image_attachments, file_attachments
                ):
                    if chunk.text or chunk.thinking:
                        has_yielded_content = True
                    yield chunk
                return  # Clean finish — no error
            except DoubaoRateLimitError as exc:
                # Never retry if we already yielded content to the caller —
                # retrying would produce duplicate/corrupted output.
                if has_yielded_content:
                    raise
                # Re-raise immediately if no handler or no verify challenge
                if not self._captcha_handler or not exc.verify_data:
                    raise
                if attempt >= self.max_captcha_retries:
                    raise
                resolved = await self._captcha_handler.handle(
                    exc.verify_data, self.fp, self.device_id
                )
                if not resolved:
                    raise

    async def chat_completion(
        self,
        text: str,
        need_deep_think: int = 0,
        bot_id: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, str]]] = None,
        file_attachments: Optional[List["UploadedFile"]] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_think_chunk: Optional[Callable[[str], None]] = None,
    ) -> CompletionResult:
        """Non-streaming wrapper around ``chat_stream_completion``."""
        result = CompletionResult()
        text_parts: List[str] = []
        think_parts: List[str] = []

        async for chunk in self.chat_stream_completion(
            text=text,
            need_deep_think=need_deep_think,
            bot_id=bot_id,
            image_attachments=image_attachments,
            file_attachments=file_attachments,
        ):
            if chunk.is_done:
                break
            if chunk.conversation_id:
                result.conversation_id = chunk.conversation_id
            if chunk.text:
                text_parts.append(chunk.text)
                if on_chunk:
                    on_chunk(chunk.text)
            if chunk.thinking:
                think_parts.append(chunk.thinking)
                if on_think_chunk:
                    on_think_chunk(chunk.thinking)
            if chunk.tool_info:
                result.tool_events.append(chunk.tool_info)
            if chunk.search_info:
                result.search_results.append(chunk.search_info)

        result.text = "".join(text_parts)
        result.thinking_text = "".join(think_parts)
        return result

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation from Doubao to keep the sidebar clean.

        Uses POST /samantha/thread/delete with the conversation_id (as thread_id)
        obtained from the SSE_ACK event during chat completion.

        Returns True if deletion succeeded, False otherwise.
        """
        if not conversation_id or conversation_id == "0":
            return False

        url = f"{self.BASE_URL}/samantha/thread/delete"
        payload = {"thread_id": str(conversation_id)}
        headers = {
            "Content-Type": "application/json",
            "x-tt-passport-csrf-token": (
                self.cookies.get("passport_csrf_token")
                or self.cookies.get("passport_csrf_token_default")
                or ""
            ),
        }

        try:
            async with self._session.post(
                url,
                params=self._security_params(),
                json=payload,
                headers=headers,
                cookies=self.cookies,
            ) as resp:
                if resp.status != 200:
                    log.warning(
                        "delete_conversation %s: HTTP %d", conversation_id, resp.status
                    )
                    return False
                data = await resp.json()
                code = data.get("code", -1)
                if code == 0:
                    log.debug("Deleted conversation %s", conversation_id)
                    return True
                log.warning(
                    "delete_conversation %s: code=%d msg=%s",
                    conversation_id, code, data.get("msg", ""),
                )
                return False
        except Exception as exc:
            log.warning("delete_conversation %s failed: %s", conversation_id, exc)
            return False

    async def generate_image(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        ref_image_key: Optional[str] = None,
    ) -> "ImageGenerationResult":
        """Generate images using Doubao's AI image generation.

        Uses /samantha/chat/completion with content_type=2009 (SamanthaImageInput)
        and skill_type=3 (SkillImageGen).

        Args:
            prompt: Text description of the image to generate.
            ratio: Aspect ratio string (e.g. "1:1", "16:9", "9:16", "4:3", "3:4").
                   If None, the server picks a default (usually 1:1).
            ref_image_key: Optional image key (from upload_image) to use as reference.

        Returns:
            ImageGenerationResult with list of generated images.
        """
        message: Dict[str, Any] = {
            "content": json.dumps({"text": prompt}, ensure_ascii=False),
            "content_type": 2009,  # SamanthaImageInput
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
                {"type": "image", "key": ref_image_key, "extra": {"refer_types": "overall"}}
            ]

        if ratio:
            message["content"] = json.dumps(
                {"text": prompt, "ratio": ratio}, ensure_ascii=False
            )

        sent_event: Dict[str, Any] = {
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

        url = (
            f"{self.BASE_URL}/samantha/chat/completion?"
            f"{urlencode(self._security_params())}"
        )
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Agw-Js-Conv": "str",
        }

        async with self.session.post(
            url,
            data=json.dumps(sent_event, ensure_ascii=False),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"generate_image failed ({resp.status}): {error_text[:500]}"
                )
            raw = (await resp.read()).decode("utf-8", errors="replace")

        self._check_gateway_error(raw)

        result = ImageGenerationResult(prompt=prompt)
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
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("event_type")
            if event_type == 2005:
                detail = data.get("event_data", "")
                raise DoubaoChatError(f"generate_image error: {str(detail)[:500]}")
            if event_type != 2001:
                continue

            event_data_str = data.get("event_data", "")
            try:
                ed = json.loads(event_data_str) if isinstance(event_data_str, str) else event_data_str
            except json.JSONDecodeError:
                continue

            msg = ed.get("message", {})
            if msg.get("content_type") != 2010:
                continue

            content_str = msg.get("content", "")
            try:
                content = json.loads(content_str) if isinstance(content_str, str) else content_str
            except json.JSONDecodeError:
                continue

            for item in content.get("data", []):
                if not isinstance(item, dict):
                    continue
                ori = item.get("image_ori", {}) or {}
                raw = item.get("image_raw", {}) or {}
                thumb = item.get("image_thumb", {}) or {}
                img = GeneratedImage(
                    key=item.get("key", ""),
                    thumb_url=thumb.get("url", ""),
                    ori_url=ori.get("url", ""),
                    raw_url=raw.get("url", ""),
                    width=ori.get("width", 0) or thumb.get("width", 0),
                    height=ori.get("height", 0) or thumb.get("height", 0),
                    format=ori.get("format", "") or thumb.get("format", ""),
                )
                result.images.append(img)

        return result

    async def generate_video(
        self,
        prompt: str,
        ratio: Optional[str] = None,
        camera_movement: Optional[str] = None,
        ref_image_key: Optional[str] = None,
        timeout: float = 300,
    ) -> "VideoGenerationResult":
        """Generate video using Doubao's AI video generation.

        Uses /samantha/chat/completion with content_type=2020
        (SamanthaVideoGenerationInput) and skill_type=17 (SkillVideoGeneration).

        Video generation is ASYNC: the initial request returns a task_id,
        then the client polls /samantha/chat/async/stream for the result.

        Args:
            prompt: Text description of the video to generate.
            ratio: Aspect ratio (e.g. "16:9", "9:16", "1:1").
            camera_movement: Camera movement style (optional).
            ref_image_key: Optional image key (from upload_image) for img2video.
            timeout: Max seconds to wait for video generation (default 300).

        Returns:
            VideoGenerationResult with list of generated videos.
        """
        content_dict: Dict[str, Any] = {"text": prompt}
        if ratio:
            content_dict["ratio"] = ratio
        if camera_movement:
            content_dict["camera_movement"] = camera_movement

        message: Dict[str, Any] = {
            "content": json.dumps(content_dict, ensure_ascii=False),
            "content_type": 2020,  # SamanthaVideoGenerationInput
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 17,
                "skill_type_no_default": 17,
                "skill_id": "17",
                "skill_id_no_default": "17",
            },
        }

        if ref_image_key:
            message["attachments"] = [
                {"type": "image", "key": ref_image_key}
            ]

        sent_event: Dict[str, Any] = {
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

        url = (
            f"{self.BASE_URL}/samantha/chat/completion?"
            f"{urlencode(self._security_params())}"
        )
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Agw-Js-Conv": "str",
        }

        # Phase 1: Submit video generation request
        async with self.session.post(
            url,
            data=json.dumps(sent_event, ensure_ascii=False),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"generate_video failed ({resp.status}): {error_text[:500]}"
                )
            raw = (await resp.read()).decode("utf-8", errors="replace")

        # Extract task_id from fin_reason in CMPL events
        task_id = self._extract_async_task_id(raw)

        if not task_id:
            # No async task — maybe error in initial response or sync result
            # Check for error messages in text content
            text_parts = []
            for block in raw.split("\n\n"):
                if not block.strip():
                    continue
                for line in block.strip().split("\n"):
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            if data.get("event_type") == 2001:
                                ed = json.loads(data["event_data"])
                                msg = ed.get("message", {})
                                if msg.get("content_type") == 2001:
                                    c = json.loads(msg["content"])
                                    text_parts.append(c.get("text", ""))
                        except (json.JSONDecodeError, KeyError):
                            pass
            full_text = "".join(text_parts)
            if "服务过载" in full_text or "重试" in full_text:
                raise DoubaoChatError(
                    f"generate_video: 服务过载，请稍后重试"
                )
            # Try parsing as sync response (content_type=2021)
            return self._parse_video_sse(raw, prompt)

        # Phase 2: Poll /samantha/chat/async/stream for video result
        return await self._poll_async_video(task_id, prompt, timeout)

    def _extract_async_task_id(self, raw: str) -> Optional[str]:
        """Extract async task_id from SSE response (fin_reason.async_task.id)."""
        self._check_gateway_error(raw)
        for block in raw.split("\n\n"):
            if not block.strip():
                continue
            for line in block.strip().split("\n"):
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                    if data.get("event_type") != 2001:
                        continue
                    ed = json.loads(data["event_data"])
                    fin_reason = ed.get("fin_reason", {})
                    if not fin_reason:
                        continue
                    if fin_reason.get("reason") == 1:  # FinReasonAsyncTask
                        async_task = fin_reason.get("async_task", {})
                        tid = async_task.get("id", "")
                        if tid:
                            return tid
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    async def _poll_async_video(
        self, task_id: str, prompt: str, timeout: float
    ) -> "VideoGenerationResult":
        """Poll /samantha/chat/async/stream SSE for video generation result."""
        # Build URL without fp param (async/stream doesn't use it)
        params = self._security_params()
        params.pop("fp", None)
        url = (
            f"{self.BASE_URL}/samantha/chat/async/stream?"
            f"{urlencode(params)}"
        )
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Agw-Js-Conv": "str",
        }
        body = json.dumps({"task_id": task_id, "event_id": 0})

        async with self.session.post(
            url,
            data=body,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"generate_video async poll failed ({resp.status}): "
                    f"{error_text[:500]}"
                )
            raw = (await resp.read()).decode("utf-8", errors="replace")

        return self._parse_video_sse(raw, prompt)

    def _parse_video_sse(
        self, raw: str, prompt: str
    ) -> "VideoGenerationResult":
        """Parse SSE response for video generation (content_type=2021)."""
        self._check_gateway_error(raw)
        result = VideoGenerationResult(prompt=prompt)
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
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("event_type")
            if event_type == 2005:
                detail = data.get("event_data", "")
                raise DoubaoChatError(
                    f"generate_video error: {str(detail)[:500]}"
                )
            if event_type != 2001:
                continue

            event_data_str = data.get("event_data", "")
            try:
                ed = (
                    json.loads(event_data_str)
                    if isinstance(event_data_str, str)
                    else event_data_str
                )
            except json.JSONDecodeError:
                continue

            msg = ed.get("message", {})
            if msg.get("content_type") != 2021:
                continue

            content_str = msg.get("content", "")
            try:
                content = (
                    json.loads(content_str)
                    if isinstance(content_str, str)
                    else content_str
                )
            except json.JSONDecodeError:
                continue

            # Video response: extract video_model similar to music
            for item in content.get("data", [content]):
                if not isinstance(item, dict):
                    continue
                # Try direct video_url field
                video_url = (
                    item.get("video_url", "")
                    or item.get("url", "")
                )
                # Or decode from video_model (base64 like music)
                if not video_url:
                    vm_str = item.get("video_model", "")
                    if vm_str:
                        try:
                            vm = (
                                json.loads(vm_str)
                                if isinstance(vm_str, str)
                                else vm_str
                            )
                            vlist = vm.get("video_list", {})
                            for _q, vinfo in vlist.items():
                                main_b64 = vinfo.get("main_url", "")
                                if main_b64:
                                    video_url = base64.b64decode(
                                        main_b64
                                    ).decode("utf-8", errors="replace")
                                    break
                        except (json.JSONDecodeError, Exception):
                            pass

                cover_url = (
                    item.get("cover_url", "")
                    or item.get("cover", {}).get("url", "")
                )
                video = GeneratedVideo(
                    video_url=video_url,
                    cover_url=cover_url,
                    width=item.get("width", 0),
                    height=item.get("height", 0),
                    duration=item.get("duration", 0.0),
                )
                if video.video_url:
                    result.videos.append(video)

        return result

    async def generate_music(
        self,
        prompt: str,
        lyric: Optional[str] = None,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        gender: Optional[str] = None,
        theme: Optional[str] = None,
        generation_type: Optional[str] = None,
    ) -> "MusicGenerationResult":
        """Generate music using Doubao's AI music generation.

        Uses /samantha/chat/completion with content_type=2005
        (SamanthaMusicGenInput) and skill_type=9 (SkillMusicGen).

        Args:
            prompt: Text description of the music to generate.
            lyric: Explicit lyrics for the song (optional).
            genre: Music genre, e.g. "pop", "rock", "jazz" (optional).
            mood: Mood/emotion, e.g. "happy", "sad" (optional).
            gender: Vocalist gender preference (optional).
            theme: Theme of the song (optional).
            generation_type: Generation mode (optional).

        Returns:
            MusicGenerationResult with list of generated tracks.
        """
        content_dict: Dict[str, Any] = {"text": prompt}
        if lyric:
            content_dict["lyric"] = lyric
        if genre:
            content_dict["genre"] = genre
        if mood:
            content_dict["mood"] = mood
        if gender:
            content_dict["gender"] = gender
        if theme:
            content_dict["theme"] = theme
        if generation_type:
            content_dict["generation_type"] = generation_type

        message: Dict[str, Any] = {
            "content": json.dumps(content_dict, ensure_ascii=False),
            "content_type": 2005,  # SamanthaMusicGenInput
            "attachments": [],
            "references": [],
            "skill": {
                "skill_type": 9,
                "skill_type_no_default": 9,
                "skill_id": "9",
                "skill_id_no_default": "9",
            },
        }

        sent_event: Dict[str, Any] = {
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

        url = (
            f"{self.BASE_URL}/samantha/chat/completion?"
            f"{urlencode(self._security_params())}"
        )
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Agw-Js-Conv": "str",
        }

        async with self.session.post(
            url,
            data=json.dumps(sent_event, ensure_ascii=False),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise DoubaoChatError(
                    f"generate_music failed ({resp.status}): {error_text[:500]}"
                )
            raw = (await resp.read()).decode("utf-8", errors="replace")

        return self._parse_music_response(raw, prompt)

    def _parse_music_response(
        self, raw: str, prompt: str
    ) -> "MusicGenerationResult":
        """Parse SSE response for music generation.

        Music response uses content_type=2006 with tasks dict structure.
        Audio URL is base64-encoded inside video_model.video_list.*.main_url.
        """
        self._check_gateway_error(raw)
        result = MusicGenerationResult(prompt=prompt)
        # We want the LAST content_type=2006 event with is_finish=True
        final_content = None
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
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = data.get("event_type")
            if event_type == 2005:
                detail = data.get("event_data", "")
                raise DoubaoChatError(
                    f"generate_music error: {str(detail)[:500]}"
                )
            if event_type != 2001:
                continue

            event_data_str = data.get("event_data", "")
            try:
                ed = (
                    json.loads(event_data_str)
                    if isinstance(event_data_str, str)
                    else event_data_str
                )
            except json.JSONDecodeError:
                continue

            msg = ed.get("message", {})
            if msg.get("content_type") not in (2006, 2004):
                continue

            content_str = msg.get("content", "")
            try:
                content = (
                    json.loads(content_str)
                    if isinstance(content_str, str)
                    else content_str
                )
            except json.JSONDecodeError:
                continue

            # Keep updating — we want the final (most complete) version
            if ed.get("is_finish") or not final_content:
                final_content = content

        if not final_content:
            return result

        # Parse tasks dict: {"0": {...}, "1": {...}}
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
            if task.get("music_gen_failed"):
                msg = task.get("music_gen_failed_msg", "unknown error")
                raise DoubaoChatError(f"generate_music failed: {msg}")

            # Extract audio URL from video_model
            audio_url = ""
            duration = 0.0
            video_model_str = task.get("video_model", "")
            if video_model_str:
                try:
                    vm = (
                        json.loads(video_model_str)
                        if isinstance(video_model_str, str)
                        else video_model_str
                    )
                    duration = vm.get("video_duration", 0.0)
                    vlist = vm.get("video_list", {})
                    for _quality, vinfo in vlist.items():
                        main_url_b64 = vinfo.get("main_url", "")
                        if main_url_b64:
                            audio_url = base64.b64decode(
                                main_url_b64
                            ).decode("utf-8", errors="replace")
                            break
                except (json.JSONDecodeError, Exception):
                    pass

            # Extract cover URL
            cover_url = ""
            cover = task.get("cover", {})
            if isinstance(cover, dict):
                cover_ori = cover.get("image_ori", {}) or {}
                cover_url = cover_ori.get("url", "")

            track = GeneratedMusic(
                audio_url=audio_url,
                title=task.get("title", ""),
                duration=duration,
                lyrics=task.get("lyric", ""),
                cover_url=cover_url,
                vid=task.get("vid", ""),
            )
            if track.audio_url or track.title:
                result.tracks.append(track)

        return result

    # ------------------------------------------------------------------
    # File upload (TOS / ImageX flow)
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
    ) -> "UploadedFile":
        """Upload a file to Doubao's storage (ByteDance TOS via ImageX proxy).

        Supported formats: PDF, TXT, DOCX, XLSX, PPTX, CSV, MD, code files,
        images, and more (see UPLOAD_SUPPORTED_EXTENSIONS).

        Flow:
          1. POST /alice/resource/prepare_upload -> STS credentials + service_id
          2. GET  /top/v1?Action=ApplyImageUpload -> upload address + auth token
          3. POST https://{tos_host}/upload/v1/{store_uri} -> upload binary
          4. POST /top/v1?Action=CommitImageUpload -> confirm upload

        Args:
            file_data: Raw file bytes.
            filename: Original filename (used for extension detection).

        Returns:
            UploadedFile with uri, name, size, file_type.
            Pass this to completion/chat methods via file_attachments param.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext and ext not in UPLOAD_SUPPORTED_EXTENSIONS:
            log.warning("File extension '%s' may not be supported", ext)

        file_size = len(file_data)
        crc32 = format(zlib.crc32(file_data) & 0xFFFFFFFF, "08x")
        params = self._security_params()

        # Step 1: Prepare upload — get STS credentials
        prepare_url = f"{self.BASE_URL}/alice/resource/prepare_upload?{urlencode(params)}"
        async with self.session.post(
            prepare_url,
            json={"tenant_id": "5", "scene_id": "5", "resource_type": 1},
        ) as resp:
            body = await resp.json()
            if body.get("code") != 0:
                raise DoubaoChatError(
                    f"prepare_upload failed: code={body.get('code')} msg={body.get('msg')}"
                )
            data = body["data"]
            service_id = data["service_id"]
            auth_token = data["upload_auth_token"]
            ak = auth_token["access_key"]
            sk = auth_token["secret_key"]
            st = auth_token["session_token"]

        # Step 2: ApplyImageUpload via /top/v1 proxy
        file_ext = f".{ext}" if ext else ""
        apply_url = (
            f"{self.BASE_URL}/top/v1?"
            f"Action=ApplyImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}&NeedFallback=true"
            f"&FileSize={file_size}&FileExtension={file_ext}"
            f"&s=jdnfglwfkl"
        )
        sign_headers = _aws_sign_v4("GET", apply_url, "", ak, sk, st)
        async with self.session.get(apply_url, headers=sign_headers) as resp:
            body = await resp.json()
            result_data = body.get("Result")
            if not result_data:
                err = body.get("ResponseMetadata", {}).get("Error", {})
                raise DoubaoChatError(
                    f"ApplyImageUpload failed: {err.get('Code')} {err.get('Message')}"
                )
            upload_addr = result_data["UploadAddress"]
            store_info = upload_addr["StoreInfos"][0]
            store_uri = store_info["StoreUri"]
            tos_auth = store_info["Auth"]
            session_key = upload_addr["SessionKey"]
            upload_hosts = upload_addr.get("UploadHosts", [])

        # Step 3: Upload binary to TOS
        tos_host = upload_hosts[0] if upload_hosts else "tos-mya2lf.vodupload.com"
        upload_url = f"https://{tos_host}/upload/v1/{store_uri}"
        upload_headers = {
            "Authorization": tos_auth,
            "Content-CRC32": crc32,
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_read=None),
        ) as tos_session:
            async with tos_session.post(
                upload_url, data=file_data, headers=upload_headers
            ) as resp:
                tos_resp = await resp.json()
                if tos_resp.get("code") != 2000:
                    raise DoubaoChatError(
                        f"TOS upload failed: {tos_resp.get('message', tos_resp)}"
                    )

        # Step 4: CommitImageUpload
        commit_url = (
            f"{self.BASE_URL}/top/v1?"
            f"Action=CommitImageUpload&Version=2018-08-01"
            f"&ServiceId={service_id}"
        )
        commit_body = json.dumps({"SessionKey": session_key})
        sign_headers2 = _aws_sign_v4("POST", commit_url, commit_body, ak, sk, st)
        sign_headers2["Content-Type"] = "application/json"
        async with self.session.post(
            commit_url, data=commit_body, headers=sign_headers2
        ) as resp:
            body = await resp.json()
            results = body.get("Result", {}).get("Results", [])
            if not results or results[0].get("UriStatus") != 2000:
                raise DoubaoChatError(
                    f"CommitImageUpload failed: {body}"
                )

        log.info("File uploaded: %s -> %s", filename, store_uri)
        return UploadedFile(
            uri=store_uri,
            name=filename,
            size=file_size,
            file_type=ext,
        )

    async def upload_image(
        self,
        image_bytes: bytes,
        filename: str = "image.png",
    ) -> Dict[str, str]:
        """Upload an image and return attachment metadata.

        Two-step flow:
          1. POST /samantha/pages/upload_image -> {uri, url}
          2. POST /alice/message/get_file_url  -> {uri, main_url (CDN)}

        Returns dict with keys: uri, cdn_url, name, format, width, height.
        """
        params = self._security_params()
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "png"

        upload_url = (
            f"{self.BASE_URL}/samantha/pages/upload_image?"
            f"{urlencode(params)}"
        )
        form = aiohttp.FormData()
        form.add_field(
            "data", image_bytes,
            filename=filename,
            content_type=f"image/{ext}",
        )
        form.add_field("file_type", ext)

        # 独立 session: 避免 self.session 默认 Content-Type: application/json 覆盖 multipart
        upload_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/135.0.0.0 Safari/537.36"
            ),
            "Origin": "https://www.doubao.com",
            "Referer": "https://www.doubao.com/chat",
            "x-tt-passport-csrf-token": (
                self.cookies.get("passport_csrf_token")
                or self.cookies.get("passport_csrf_token_default")
                or ""
            ),
        }
        async with aiohttp.ClientSession(
            cookies=self.cookies,
            headers=upload_headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as upload_session:
            async with upload_session.post(upload_url, data=form) as resp:
                if resp.status != 200:
                    raise DoubaoChatError(
                        f"Image upload failed ({resp.status}): "
                        f"{(await resp.text())[:500]}"
                    )
                body = await resp.json()
                if body.get("code") != 0:
                    raise DoubaoChatError(
                        f"Image upload error: {body.get('msg', body)}"
                    )
                upload_data = body["data"]
                uri_short = upload_data["uri"]

        file_url_endpoint = (
            f"{self.BASE_URL}/alice/message/get_file_url?"
            f"{urlencode(params)}"
        )
        async with self.session.post(
            file_url_endpoint,
            json={
                "uris": [uri_short],
                "type": "image",
                "format": ext,
                "expire_second": 3600,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                raise DoubaoChatError(
                    f"get_file_url failed ({resp.status}): "
                    f"{(await resp.text())[:500]}"
                )
            body = await resp.json()
            if body.get("code") != 0:
                raise DoubaoChatError(
                    f"get_file_url error: {body.get('msg', body)}"
                )
            data = body.get("data", {})
            file_urls = data.get("file_urls", []) if isinstance(data, dict) else []
            if not file_urls:
                raise DoubaoChatError("get_file_url returned no file_urls")
            file_info = file_urls[0]

        return {
            "uri": file_info["uri"],
            "cdn_url": file_info["main_url"],
            "name": filename,
            "format": ext,
            "width": "64",
            "height": "64",
        }

    async def get_file_download_url(
        self,
        uri: str,
        expire_seconds: int = 3600,
    ) -> str:
        """Get a temporary CDN download URL for a previously uploaded file.

        Args:
            uri: TOS URI from upload_file() or upload_image().
            expire_seconds: URL validity period (default 1 hour, max unknown).

        Returns:
            Public CDN URL string for downloading the file.
        """
        params = self._security_params()
        ext = uri.rsplit(".", 1)[-1] if "." in uri else ""
        file_url_endpoint = (
            f"{self.BASE_URL}/alice/message/get_file_url?"
            f"{urlencode(params)}"
        )
        async with self.session.post(
            file_url_endpoint,
            json={
                "uris": [uri],
                "type": "file",
                "format": ext,
                "expire_second": expire_seconds,
            },
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                raise DoubaoChatError(
                    f"get_file_url failed ({resp.status}): "
                    f"{(await resp.text())[:500]}"
                )
            body = await resp.json()
            if body.get("code") != 0:
                raise DoubaoChatError(
                    f"get_file_url error: {body.get('msg', body)}"
                )
            data = body.get("data", {})
            file_urls = data.get("file_urls", []) if isinstance(data, dict) else []
            if not file_urls:
                raise DoubaoChatError("get_file_url returned no file_urls")
            return file_urls[0].get("main_url", "")

    @classmethod
    def from_session(
        cls,
        session_file: str = ".doubao_session.json",
        bot_id: str = DEFAULT_BOT_ID,
        timeout_seconds: int = 120,
        captcha_handler: Union[None, str, "CaptchaHandler"] = "auto",
        max_captcha_retries: int = 3,
    ) -> "DoubaoChatClient":
        """
        Create a client from a saved ``.doubao_session.json`` file.

        The file is written by the GUI (QR login or Chromium load) and
        contains cookies, device params, and the ``fp_verified`` flag.

        Args:
            session_file: Path to the session JSON file.
            captcha_handler: See ``__init__`` for details.
        """
        from .session import load_session

        session = load_session(session_file)
        cookies = session["cookies"]
        params = session["params"]

        if not cookies.get("sessionid"):
            raise DoubaoChatError(
                f"No sessionid in {session_file}. "
                "Use QR login or provide cookies via /v1/session/update."
            )

        # msToken is optional — API works without it, and fake values
        # trigger rate limiting (710022002). Empty string is safest.
        ms_token = cookies.get("msToken", "")
        device_id = params.get("device_id") or "714003710229497"
        web_id = params.get("web_id") or "7604137868021548590"
        fp = params.get("fp") or "verify_mlcfw5f7_TPq0YmFD_NrsC_4RuQ_BJPg_M5W7i58I7wV0"

        return cls(
            cookies=cookies,
            ms_token=ms_token,
            device_id=device_id,
            web_id=web_id,
            fp=fp,
            bot_id=bot_id,
            timeout_seconds=timeout_seconds,
            captcha_handler=captcha_handler,
            max_captcha_retries=max_captcha_retries,
        )
