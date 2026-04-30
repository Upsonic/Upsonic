---
name: agent-frontend-interfaces
description: Use when exposing an Upsonic Agent or AutonomousAgent to end-users via Slack, WhatsApp, Telegram, Discord, Gmail, generic SMTP/IMAP Mail, or built-in WebSocket channels. Use when a user asks to wire a chat/email/messaging frontend to an agent, mount channels on a FastAPI app via InterfaceManager, configure Bearer auth, handle webhook signature verification, dedup, whitelisting, reset commands, TASK vs CHAT mode, streaming replies, HITL confirmation buttons, media/attachment ingestion, or AutonomousAgent heartbeat ticks. Trigger when the user mentions Interface, InterfaceManager, InterfaceMode, InterfaceSettings, InterfaceResetCommand, SlackInterface, WhatsAppInterface, TelegramInterface, DiscordInterface, GmailInterface, MailInterface, WebSocketManager, attach_routes, /reset, signing_secret, X-Hub-Signature-256, X-Telegram-Bot-Api-Secret-Token, Discord Gateway, IMAP polling, heartbeat_channel, or per-channel allowed_user_ids/allowed_numbers/allowed_emails.
---

# `src/upsonic/interfaces/` — Agent-Frontend Channel Integrations

## 1. What this folder is

The `interfaces/` package is Upsonic's **agent-frontend layer**: a unified
adapter framework that lets a single `Agent` (or `AutonomousAgent`) be exposed
to end-users through real-world communication channels — chat apps, email,
WebSockets, and any future plugin a developer wants to write.

Every concrete interface (Slack, WhatsApp, Telegram, Discord, Gmail, generic
SMTP/IMAP `Mail`) is a subclass of the abstract `Interface` base class. Each
one:

- Receives inbound messages over a transport native to that channel
  (HTTP webhook, WebSocket Gateway, IMAP polling).
- Translates the channel payload into a generic `Task` (TASK mode) or feeds
  it into a per-user `Chat` session (CHAT mode).
- Calls into the agent (`agent.do_async` / `chat.invoke` / `agent.astream`)
  and ships the response back over the channel native send-API.

A central `InterfaceManager` mounts one or more interfaces onto a single
FastAPI app, exposes them on shared core routes (`/health`, `/ws/{client_id}`,
`/`), enforces optional Bearer-token auth, and serves everything via uvicorn
through a single `serve()` call.

If you remove every interface implementation, what's left in this folder is
the *plumbing* — base class, schemas, settings, manager, websocket
multiplexer, auth — i.e. the contract every channel adapter must satisfy.

| Channel | Transport | Access control field | Reset cmd | Stream support |
|---------|-----------|----------------------|-----------|----------------|
| Slack | HTTP webhook (`/slack/events`) signed with HMAC-SHA256 | `allowed_user_ids` | `/reset` | Yes |
| WhatsApp | Meta webhook (`GET`+`POST /whatsapp/webhook`), `X-Hub-Signature-256` | `allowed_numbers` | `/reset` | No |
| Telegram | Bot API webhook (`/telegram/webhook`), `X-Telegram-Bot-Api-Secret-Token` | `allowed_user_ids` | `/reset` | Yes |
| Discord | Gateway WebSocket + optional `/discord/interaction` HTTP endpoint | `allowed_user_ids` / `allowed_channel_ids` / `allowed_guild_ids` | `/reset` | Yes |
| Gmail | Polling via Google API (`POST /gmail/check`) | `allowed_emails` | `/reset` | No |
| Mail (SMTP/IMAP) | IMAP polling + heartbeat auto-poll (`POST /mail/check`) | `allowed_emails` | `/reset` | No |
| WebSocket (built-in) | `/ws/{client_id}` on the manager itself | Bearer token via auth message | n/a | n/a |

## 2. Folder layout

```
src/upsonic/interfaces/
├── __init__.py            # Public API + lazy class loaders
├── auth.py                # Bearer-token auth dependency + WS token validator
├── base.py                # Interface ABC, mode handling, Chat session cache
├── manager.py             # InterfaceManager: FastAPI app, lifespan, routes, serve()
├── schemas.py             # InterfaceMode, ResetCommand, HealthCheck/Error/WS schemas
├── settings.py            # InterfaceSettings (BaseSettings, env prefix UPSONIC_INTERFACE_)
├── websocket_manager.py   # WebSocketConnection + WebSocketManager (broadcast/auth/cleanup)
│
├── slack/
│   ├── __init__.py        # Lazy export SlackInterface, SlackEventResponse, SlackChallengeResponse
│   ├── slack.py           # SlackInterface (HMAC verify, dedup, threads, italics)
│   └── schemas.py         # SlackEventResponse, SlackChallengeResponse
│
├── whatsapp/
│   ├── __init__.py        # Lazy export WhatsAppInterface, WhatsAppWebhookPayload
│   ├── whatsapp.py        # WhatsAppInterface (Meta webhook, media, image-out)
│   └── schemas.py         # WhatsAppWebhookPayload + nested envelope models
│
├── telegram/
│   ├── __init__.py        # Eager export TelegramInterface + every schema
│   ├── telegram.py        # TelegramInterface (webhook, callbacks, all media kinds, HITL)
│   └── schemas.py         # 20+ Pydantic models mirroring Bot API
│
├── discord/
│   ├── __init__.py        # Eager export DiscordInterface + schemas
│   ├── discord.py         # DiscordInterface (Gateway WS + interactions endpoint, HITL buttons)
│   └── schemas.py         # DiscordMessage/User/Guild/Interaction etc.
│
├── gmail/
│   ├── __init__.py        # Lazy export GmailInterface, CheckEmailsResponse
│   ├── gmail.py           # GmailInterface (poll-driven, replies via GmailTools)
│   └── schemas.py         # CheckEmailsResponse, AgentEmailResponse
│
└── mail/
    ├── __init__.py        # Lazy export MailInterface + response schemas
    ├── mail.py            # MailInterface (generic SMTP/IMAP, heartbeat auto-poll)
    └── schemas.py         # EmailSummary, SendEmailRequest, SearchEmailRequest, etc.
```

## 3. Top-level files

### 3.1 `__init__.py` — Public API surface

The top-level package uses **PEP 562 lazy `__getattr__`** to defer importing
the heavyweight channel modules until first access. This matters because each
channel pulls in its own SDK / HTTP client (e.g. Discord pulls `websockets`,
Gmail pulls Google libs).

| Name exported | Source module | Purpose |
|---------------|---------------|---------|
| `Interface` | `.base` | ABC every channel subclasses |
| `InterfaceManager` | `.manager` | Multiplex + serve interfaces |
| `InterfaceMode` | `.schemas` | `TASK` / `CHAT` enum |
| `InterfaceResetCommand` | `.schemas` | Match `/reset`-style commands |
| `InterfaceSettings` | `.settings` | FastAPI/uvicorn config |
| `WhatsAppInterface` (alias `Whatsapp`) | `.whatsapp.whatsapp` | Meta WhatsApp Business |
| `SlackInterface` (alias `Slack`) | `.slack.slack` | Slack Events API |
| `GmailInterface` (alias `Gmail`) | `.gmail.gmail` | Gmail polling |
| `TelegramInterface` (alias `Telegram`) | `.telegram.telegram` | Telegram Bot |
| `DiscordInterface` (alias `Discord`) | `.discord.discord` | Discord Gateway |
| `MailInterface` (alias `Mail`) | `.mail.mail` | SMTP/IMAP |
| `WebSocketManager`, `WebSocketConnection` | `.websocket_manager` | Built-in real-time channel |
| `get_authentication_dependency`, `validate_websocket_token` | `.auth` | Auth helpers |
| Schemas: `HealthCheckResponse`, `ErrorResponse`, `WhatsAppWebhookPayload`, `TelegramWebhookPayload`, `DiscordGatewayPayload`, `WebSocketMessage`, `WebSocketConnectionInfo`, `WebSocketStatusResponse` | various | Public response models |

`__version__ = "1.0.0"` is also exposed.

### 3.2 `auth.py` — Bearer-token authentication

Two helpers, both keyed off `InterfaceSettings.security_key`:

```python
security = HTTPBearer(auto_error=False)

def get_authentication_dependency(settings: InterfaceSettings):
    async def auth_dependency(credentials = Depends(security)) -> bool:
        if not settings.is_auth_enabled():
            return True
        if not credentials:
            raise HTTPException(401, "Authorization header required",
                                headers={"WWW-Authenticate": "Bearer"})
        if credentials.credentials != settings.security_key:
            raise HTTPException(401, "Invalid authentication token",
                                headers={"WWW-Authenticate": "Bearer"})
        return True
    return auth_dependency

def validate_websocket_token(token, settings) -> bool:
    if not settings.is_auth_enabled():
        return True
    return bool(token) and token == settings.security_key
```

Auth is **opt-in by absence**: setting `UPSONIC_INTERFACE_SECURITY_KEY=<token>`
turns it on. Otherwise both HTTP and WS routes are open. The same key value
is used for both transports.

### 3.3 `schemas.py` — Cross-channel schemas

| Model | Role |
|-------|------|
| `InterfaceMode(str, Enum)` | `"task"` (stateless, default) or `"chat"` (stateful per user) |
| `InterfaceResetCommand` | Holds `command="/reset"`, `case_sensitive=False`, plus `matches(text)` |
| `HealthCheckResponse` | Returned by `GET /health` — status, timestamp, per-iface dict, WS conn count |
| `ErrorResponse` | `error`, `detail`, `timestamp`; produced by middleware/handlers |
| `WebSocketMessage` | Generic WS envelope: `type`, `data`, `timestamp`, `connection_id` |
| `WebSocketConnectionInfo` | One row per active WS connection (id, name, client_id, connected_at, metadata) |
| `WebSocketStatusResponse` | `total_connections` + list of `WebSocketConnectionInfo` |

### 3.4 `settings.py` — `InterfaceSettings`

A `pydantic_settings.BaseSettings` with `env_prefix='UPSONIC_INTERFACE_'`.
Default values cover normal local-dev:

| Setting | Default | Notes |
|---------|---------|-------|
| `security_key` | `None` | Set to enable Bearer auth on all manager routes |
| `app_title` / `app_description` / `app_version` | "Upsonic Interface Manager" / ... / "1.0.0" | Surfaced in OpenAPI |
| `debug` | `False` | When True, exception handler returns `detail=str(exc)` |
| `docs_url`, `redoc_url`, `openapi_url` | `/docs`, `/redoc`, `/openapi.json` | Set to `None` to disable |
| `cors_enabled` + `cors_origins`/`cors_allow_credentials`/`cors_allow_methods`/`cors_allow_headers` | `True` + `["*"]` everywhere | Mounted as `CORSMiddleware` |
| `trusted_hosts` | `None` | If a list is provided, mounts `TrustedHostMiddleware` |
| `max_upload_size` | `10 * 1024 * 1024` | Custom middleware returns 413 if `Content-Length` exceeds |
| `request_timeout` | `300` | Wraps `call_next` in `asyncio.wait_for` → 504 on overflow |
| `websocket_ping_interval` / `websocket_ping_timeout` | `20.0`, `20.0` | Stored in connection metadata, not enforced by uvicorn directly |
| `log_level` | `INFO` | Forwarded to uvicorn |
| `access_log` | `False` | Forwarded to uvicorn |

`is_auth_enabled()` returns `security_key is not None and len(...) > 0`.

### 3.5 `base.py` — `Interface` ABC

```python
UNAUTHORIZED_MESSAGE = "This operation not allowed"  # not configurable

class Interface(ABC):
    def __init__(self, agent, name=None, id=None,
                 mode=InterfaceMode.TASK, reset_command="/reset",
                 storage: Optional[Storage]=None): ...

    @abstractmethod
    def attach_routes(self) -> APIRouter: ...

    async def health_check(self) -> Dict[str, Any]: ...
    def get_id(self) -> str: ...
    def get_name(self) -> str: ...
    def get_mode(self) -> InterfaceMode: ...
    def is_task_mode(self) -> bool: ...
    def is_chat_mode(self) -> bool: ...
    def is_reset_command(self, text: str) -> bool: ...

    def get_chat_session(self, user_id: str) -> Chat: ...
    async def aget_chat_session(self, user_id: str) -> Chat: ...
    def reset_chat_session(self, user_id: str) -> bool: ...
    async def areset_chat_session(self, user_id: str) -> bool: ...
    def has_chat_session(self, user_id: str) -> bool: ...
    def get_all_chat_sessions(self) -> Dict[str, Chat]: ...
    def get_unauthorized_message(self) -> str: ...
```

Key responsibilities of the base class:

1. **Identity** — every interface gets a UUID `id` (or accepts one), and a
   human name (defaults to class name).
2. **Mode normalization** — accepts `InterfaceMode` enum or `"task"`/`"chat"`
   string. Stored as `self.mode`.
3. **Reset command** — wires an `InterfaceResetCommand` so subclasses just
   call `self.is_reset_command(text)` regardless of channel.
4. **Per-user chat session cache** — `self._chat_sessions: Dict[str, Chat]`.
   On first `aget_chat_session(user_id)`, the base lazily creates a
   `upsonic.chat.Chat` with:
   - `session_id = f"{self.name.lower()}_{user_id}"`
   - `full_session_memory=True`, `summary_memory=False`,
     `user_analysis_memory=False`
   - The provided `storage` (or in-memory if `None`).
5. **Storage lazy-init** — `_get_storage()` defaults to `InMemoryStorage`
   if no backend was passed.
6. **Authorization-message constant** — `UNAUTHORIZED_MESSAGE = "This
   operation not allowed"`, exposed via `get_unauthorized_message()`. The
   docstring states it is intentionally fixed.

The single abstract method is `attach_routes()` — every concrete interface
returns its own `APIRouter` that the manager mounts on the shared FastAPI app.

### 3.6 `manager.py` — `InterfaceManager`

The manager is the entry point developers actually run. Construction wires
up:

1. A FastAPI app via `_create_app()`:
   - `lifespan` context manager logs startup / closes all WS connections on
     shutdown.
   - **Request-ID middleware** — assigns a UUID to `request.state.request_id`
     and echoes it in `X-Request-ID`.
   - **CORS middleware** — gated on `settings.cors_enabled`.
   - **TrustedHost middleware** — gated on `settings.trusted_hosts`.
   - **Validate-request middleware** — checks `Content-Length` against
     `max_upload_size` (413) and wraps the handler in
     `asyncio.wait_for(timeout=settings.request_timeout)` (504).
   - **Global exception handler** — converts unhandled exceptions to
     `ErrorResponse` 500s, includes `detail` only in `debug` mode.
2. `_attach_interface_routes()` — for each interface in `self.interfaces`
   call `interface.attach_routes()` and `app.include_router(router)`.
3. `_add_core_routes()`:

| Route | Method | Auth | Purpose |
|-------|--------|------|---------|
| `/` | GET | open | Service banner: title/version/interfaces/status |
| `/health` | GET | open | Aggregated health: per-iface `health_check()` + WS connection count |
| `/ws/{client_id}` | WS | message-based | Bidirectional channel (auth → ping/message) |
| `/ws/status` | GET | Bearer | List of active WS connections |

The WebSocket endpoint is itself a chat-style mini-protocol:

```jsonc
// Server → client immediately after accept
{"event": "connected", "message": "...", "requires_auth": true, "connection_id": "abc"}
// Client → server (only if auth enabled)
{"action": "authenticate", "token": "<security_key>"}
// Server → client
{"event": "authenticated", "message": "..."}
// Then any of:
{"action": "ping"}                           → {"event": "pong", "timestamp": "..."}
{"action": "message", "content": "hello"}    → {"event": "message", "content": "hello", "client_id": "abc"}
```

`InterfaceManager` also supports dynamic management:

```python
manager.add_interface(iface)        # mount routes after construction
manager.remove_interface(name)      # only removes from list (FastAPI cannot un-mount routers)
manager.get_interface(name)         # lookup
manager.get_app() -> FastAPI        # for ASGI-server frameworks
manager.serve(host="localhost", port=7777, reload=False, workers=None,
              access_log=False, **kwargs)  # uvicorn.Server.run()
```

Logging at startup prints the bound host/port, registered interfaces, and
whether auth is enabled.

### 3.7 `websocket_manager.py` — `WebSocketManager` + `WebSocketConnection`

`WebSocketConnection` wraps a single `fastapi.WebSocket`:

- Assigns a UUID `id`, a human `name` (defaults to `connection_id`), the
  client-supplied `connection_id` (URL path), `connected_at` timestamp,
  arbitrary `metadata` dict.
- Async `send_text(data)` / `send_json(data)` swallow exceptions and return
  bool.
- `close(code, reason)` swallows on best-effort.

`WebSocketManager` provides the broker:

| Method | Purpose |
|--------|---------|
| `await connect(websocket, connection_id, metadata, requires_auth=True)` | Accept + register, send "connected" JSON event |
| `await disconnect(connection_id)` | Look up by id, close, drop from both maps |
| `await authenticate_websocket(ws)` | Mark `authenticated_connections[ws] = True`, send "authenticated" event |
| `is_authenticated(ws) -> bool` | Lookup |
| `await disconnect_websocket(ws)` | Find every `connection_id` whose ws matches, close + clean up |
| `get_all_connections() -> List[WebSocketConnection]` | For `/ws/status` |
| `get_connection_count() -> int` | For `/health` |
| `await close_all()` | Used by lifespan shutdown |

Two dicts back the broker: `connections: Dict[str, WebSocketConnection]`
keyed by client-supplied connection ID, and
`authenticated_connections: Dict[WebSocket, bool]` keyed by raw socket. An
`asyncio.Lock` guards mutating the connection map.

## 4. Subfolders walked through

### 4.1 `slack/` — `SlackInterface`

Located at `slack/slack.py`. Subclasses `Interface`. Important kwargs:

| Arg | Default | Effect |
|-----|---------|--------|
| `signing_secret` | `os.getenv("SLACK_SIGNING_SECRET")` | HMAC verification of every webhook |
| `verification_token` | `os.getenv("SLACK_VERIFICATION_TOKEN")` | Legacy, retained |
| `reply_to_mentions_only` | `True` | If True, only `app_mention` and DMs (`channel_type == "im"`) get a reply |
| `mode` | `TASK` | Switch to `CHAT` for per-user `Chat` sessions |
| `reset_command` | `"/reset"` | Disable with `None` |
| `storage` | `None` | Backing store for chat sessions |
| `allowed_user_ids` | `None` | Whitelist of Slack `U…` IDs |
| `stream` | `False` | If True, stream agent tokens by editing the message |
| `heartbeat_channel` | `None` | Where `AutonomousAgent` heartbeat sends its tick output |

Internals worth knowing:

- `_verify_slack_signature(body, timestamp, signature)`: rejects timestamps
  more than 5 minutes off, builds `v0:{ts}:{body}` and HMAC-SHA256s with the
  signing secret, compares with `hmac.compare_digest`.
- `_processed_events: Dict[str, float]` — uses `event_ts` for dedup,
  5-minute window, autocleans when size > 1000.
- `_send_slack_message(channel, thread_ts, message, italics=False)` —
  splits anything over 4000 chars into `[i/N] …` batches; in italic mode
  wraps each line as `_line_`; uses `asend_message_thread` if `thread_ts`
  is set, else `asend_message`.
- `_stream_to_slack(channel, thread_ts, stream_iterator)` — sends initial
  message, then re-edits with `slack_tools.update_message` every 0.5s as
  tokens accumulate; final flush at end.
- `_process_slack_event(event)` — dedup, mention-only filter, removes
  `<@BOT_ID>` from `app_mention` text, captures channel for auto-heartbeat,
  enforces whitelist, triggers reset, then dispatches to `_process_event_task_mode`
  or `_process_event_chat_mode`.
- Heartbeat: `_heartbeat_loop()` only runs if `agent` is `AutonomousAgent`
  with `heartbeat=True`; `period_seconds = agent.heartbeat_period * 60`;
  uses `_resolve_heartbeat_channel()` which prefers the explicit
  `heartbeat_channel` else the auto-detected one.

Routes (`prefix=/slack`):

| Method | Path | Body / response | Purpose |
|--------|------|-----------------|---------|
| POST | `/events` | Slack envelope | Verifies signature → handles `url_verification` (returns `SlackChallengeResponse`) → dispatches `event` to `BackgroundTasks` |
| GET | `/health` | dict | Channel-specific health |
| (startup) | — | — | Starts heartbeat task |

### 4.2 `whatsapp/` — `WhatsAppInterface`

Located at `whatsapp/whatsapp.py`. Implements the Meta-WhatsApp two-step
webhook handshake plus media in/out and image generation.

Distinctive methods:

- `_validate_webhook_signature(payload, signature)` — checks
  `X-Hub-Signature-256` HMAC against `app_secret`. Returns True if
  `app_secret` is unset (skip validation in dev). Constant-time compare.
- `_normalize_phone_number(phone)` — strips everything non-digit. Used to
  build the `_allowed_numbers` whitelist set.
- `_send_whatsapp_message(recipient, message, italics=False)` — splits at
  4000 chars (limit ~4096), `[i/N]` prefix, `_line_` for italics.
- `_process_image_outputs(recipient)` — pulls
  `agent.get_run_output().get_last_model_response().images`, base64-decodes
  bytes if needed, uploads each via `upload_media_async`, sends each via
  `send_image_message_async`. Text is sent separately afterwards.
- `_process_message_with_agent(message_text, sender, media_attachments)` —
  TASK mode entry. Saves any `media_attachments` (`{"image": {"bytes": ...,
  "mime_type": ...}}`) into `tempfile.NamedTemporaryFile`, builds a `Task`
  with `attachments=[temp_paths]`, runs `agent.do_async(task)`. If the agent
  raises mid-media, calls `_get_format_error_message(mime_type, e)` and
  sends the friendly error. Always cleans temp files in a `finally`.
- `_process_message_chat_mode(...)` — same flow but via `chat.invoke(task)`
  using `aget_chat_session(sender)`.
- `_process_media_message(message, sender, message_id, media_type_key)` —
  generic handler for `image`/`audio`/`video`/`document`. Calls
  `get_media_async(media_id)` to pull the bytes, picks a default caption per
  type ("Describe the image", "Reply to audio", …), wraps into
  `media_attachments`, dispatches by mode.
- `_process_message(message)` — top-level router by `message["type"]`:
  text → `_process_text_message`, `image|audio|video|document` →
  `_process_media_message`. Captures `_auto_heartbeat_recipient` once.

Routes (`prefix=/whatsapp`):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/webhook` | Meta verification: validates `hub.mode == "subscribe"` and `hub.verify_token == self.verify_token`, returns `hub.challenge` as plain text |
| POST | `/webhook` | Validates signature, parses `WhatsAppWebhookPayload`, dispatches each `message` to background, logs `statuses` |
| GET | `/health` | Channel health |
| (startup) | — | Heartbeat |

### 4.3 `telegram/` — `TelegramInterface`

The most feature-rich channel. Located at `telegram/telegram.py`. Highlights:

- Constructor accepts the full Telegram Bot API surface area:
  `parse_mode`, `disable_web_page_preview`, `disable_notification`,
  `protect_content`, `reply_in_groups`, `reply_in_channels`,
  `process_edited_messages`, `process_callback_queries`, `typing_indicator`,
  `max_message_length=4096`, `stream`, `webhook_url`, `webhook_secret`,
  `heartbeat_chat_id`.
- **Auto-set webhook**: if `webhook_url` is provided (or `TELEGRAM_WEBHOOK_URL`
  env), the startup hook calls
  `telegram_tools.aset_webhook(url=f"{webhook_url}/telegram/webhook",
  secret_token=...)`.
- **Webhook secret verification**: rejects requests where
  `X-Telegram-Bot-Api-Secret-Token != self.webhook_secret`.
- **Update routing** in `_process_update`: dispatches `message` /
  `edited_message` / `channel_post` / `edited_channel_post` /
  `callback_query` based on the corresponding feature flags.
- `_process_message` then routes by content type — `text`, `photo`,
  `document`, `voice`, `audio`, `video`, `video_note`, `sticker`,
  `location`, `venue`, `contact`, `poll`. Non-textual messages without
  binary content (sticker emoji, location coords, venue strings, contact
  fields, poll question + options) are converted into a synthetic English
  text prompt.
- **HITL confirmations** — when an agent run pauses with
  `pause_reason == "confirmation"`, `_send_confirmation_and_store` posts a
  message with two `inline_keyboard` buttons (`callback_data="cfm:<key>:0:y"`
  and `…:n`), persists the run state in `_pending_confirmations[key]`. On
  click, `_process_callback_query` extracts the key, calls
  `first_req.confirm()` or `.reject()` on the active requirement, then
  resumes via `agent.continue_run_async(run_id, requirements)`.
- **Streaming** — `_stream_to_telegram` posts an initial message then
  `editMessageText`s every 1.0s, with a final flush.
- **Media handling** — every binary kind goes through
  `_process_media_with_agent`: write bytes to a `NamedTemporaryFile` with
  the right extension, build `Task(description=caption, attachments=[path])`,
  run via TASK or CHAT, send result, friendly error via
  `_get_format_error_message` if the model can't handle the format,
  guaranteed temp-file cleanup.
- **Heartbeat** — `_resolve_heartbeat_chat_id()` prefers the explicit
  `heartbeat_chat_id` else the first chat ID seen.

Routes (`prefix=/telegram`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/webhook` | Telegram update webhook (background-dispatched) |
| POST | `/set-webhook` | Imperative webhook installer |
| POST | `/delete-webhook` | Remove webhook |
| GET | `/webhook-info` | `getWebhookInfo` proxy |
| GET | `/health` | Channel health |
| (startup) | — | Auto-set webhook + start heartbeat |

### 4.4 `discord/` — `DiscordInterface`

Located at `discord/discord.py`. Different from the others because the
primary transport is a **persistent Discord Gateway WebSocket**, not an
inbound webhook.

Gateway opcodes used: `0=DISPATCH`, `1=HEARTBEAT`, `2=IDENTIFY`, `6=RESUME`,
`7=RECONNECT`, `9=INVALID_SESSION`, `10=HELLO`, `11=HEARTBEAT_ACK`. Default
intents are `INTENT_GUILDS | INTENT_GUILD_MESSAGES | INTENT_DIRECT_MESSAGES
| INTENT_MESSAGE_CONTENT`.

Lifecycle (in `attach_routes()`):

```python
@router.on_event("startup")
async def start_gateway():
    await self._start_gateway()  # asyncio.create_task(self._gateway_connect())

@router.on_event("startup")
async def start_heartbeat():
    self._start_heartbeat()

@router.on_event("shutdown")
async def stop_gateway():
    await self._stop_gateway()
```

`_gateway_connect()` runs forever:
1. Calls `discord_tools._api_request("GET", "/gateway/bot")` to discover the
   gateway URL (falls back to `wss://gateway.discord.gg/?v=10&encoding=json`).
2. Opens the WS via `websockets.connect`.
3. Reads the HELLO frame, captures `heartbeat_interval`.
4. Spawns `_gateway_heartbeat_loop` (sends `{"op": 1, "d": <seq>}` every
   `heartbeat_interval` seconds).
5. Sends IDENTIFY (with `intents` and bot token) or RESUME (if it has a
   `session_id` and `sequence`).
6. Calls `_gateway_listen(ws)` — for each frame, dispatches by opcode:
   - `0` (`DISPATCH`) → `_process_gateway_event(event_name, data)` which
     handles `READY`, `RESUMED`, `MESSAGE_CREATE`, `INTERACTION_CREATE`.
   - `1` echoes back a heartbeat, `7` triggers reconnect, `9` clears or
     keeps the session and reconnects, `11` is the heartbeat ack.
7. On exception, sleeps 5 s and retries (logged).

Inbound messages are validated as `DiscordMessage`. The interface ignores
its own bot messages, applies `process_dm` / `process_guild_messages`
filters, then `allowed_user_ids` / `allowed_channel_ids` /
`allowed_guild_ids` whitelists. Auto-detects a heartbeat channel from the
first message seen.

Other notable behavior:

- **Typing indicator** — `_start_typing_indicator(channel_id)` spawns a
  loop calling `discord_tools.atrigger_typing(channel_id)` every 8 s
  (Discord typing expires at 10 s). `_stop_typing_indicator` cancels.
- **HITL** — same pattern as Telegram but with Discord components: action
  row containing two buttons (`style: 3 SUCCESS`, `style: 4 DANGER`,
  `custom_id="cfm:{key}:0:y|n"`). Replies use Discord's
  `acreate_interaction_response(interaction.id, interaction.token, type=6)`
  for deferred update.
- **Streaming** — initial message via `asend_message`, then
  `aedit_message` every 2 s.
- **Attachments** — pulled via `httpx.AsyncClient().get(attachment.url)`,
  written to temp file, `Task(description=caption, attachments=[path])`.

Routes (`prefix=/discord`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/interaction` | Optional HTTP Interactions endpoint (for slash cmds outside Gateway). Echoes type 1 ping, returns deferred type 5 |
| GET | `/health` | Channel health, includes `gateway_connected` flag |
| (startup) | — | Start gateway + heartbeat |
| (shutdown) | — | Cleanly close gateway |

### 4.5 `gmail/` — `GmailInterface`

Located at `gmail/gmail.py`. Pulls instead of webhooks. Routes are minimal:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/check?count=N` (header `X-Upsonic-Gmail-Secret`) | Run `check_and_process_emails(count)` |
| GET | `/health` | Health (auth + connected flags) |

`check_and_process_emails(count)`:

1. Calls `gmail_tools.get_unread_messages_raw(count)` in a thread.
2. For each message: extract `sender`, `subject`, `body`.
3. Whitelist via `is_email_allowed(sender)` (uses `_extract_sender_id`
   regex `<([^>]+)>`).
4. Reset command via `is_reset_command(body)` (CHAT mode only).
5. Dispatch to `_process_email_task_mode` (Pydantic `AgentEmailResponse`
   forced via `Task(..., response_format=AgentEmailResponse)`) or
   `_process_email_chat_mode` (per-sender `Chat`).
6. Mark read via `gmail_tools.amark_email_as_read(msg_id)`.
7. Return `CheckEmailsResponse(status, processed_count, message_ids)`.

In TASK mode the agent must return the structured `AgentEmailResponse`
(`action ∈ {"reply","ignore"}`, `reply_body`, `reasoning`). The interface
sends a reply only when `action == "reply"` and `reply_body` is non-empty.

### 4.6 `mail/` — `MailInterface`

Generic SMTP/IMAP version of Gmail (works with Outlook, Yahoo, Zoho,
self-hosted). Located at `mail/mail.py`. Adds a heartbeat **auto-poller**
on top of the manual `/mail/check` endpoint.

Routes (`prefix=/mail`, all gated by `X-Upsonic-Mail-Secret`):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/check?count=N` | Process unread emails through agent |
| GET | `/inbox?count&mailbox` | List recent emails |
| GET | `/unread?count&mailbox` | List unread |
| POST | `/send` | Send via `MailTools.asend_email` |
| POST | `/search` | IMAP search query |
| GET | `/folders` | List mailboxes |
| GET | `/status?mailbox` | Total/unseen/recent counts |
| POST | `/{uid}/read` | Mark read |
| POST | `/{uid}/unread` | Mark unread |
| POST | `/{uid}/delete` | Delete |
| POST | `/{uid}/move?destination&source` | Move folder |
| GET | `/health` | Health |
| (startup) | — | Start heartbeat auto-poller |

`check_and_process_emails(count)`:
- `aget_unread_emails(count, self.mailbox)` from `MailTools`.
- Dedup on UID (`_processed_emails` with 300 s TTL, GC at 1000 entries).
- Whitelist via `is_email_allowed` → `_extract_sender_id` regex.
- Reset command in CHAT mode (matched on `body.strip()`).
- TASK mode: enforces `Task(task_description, response_format=AgentEmailResponse,
  attachments=temp_files)` so the agent must produce the structured
  reply/ignore decision. Reply via `_send_reply` →
  `mail_tools.asend_reply(to, subject, body, message_id, references)`
  (preserves email threading).
- CHAT mode: `chat.invoke(chat_message, attachments=temp_files)`.
- Attachments: `mail_tools.aget_raw_attachments(uid, mailbox)` →
  `tempfile.NamedTemporaryFile` with extension from filename or mime guess.
- Mark read after success.

Heartbeat for `MailInterface` is **the auto-poll itself** — every
`agent.heartbeat_period * 60` seconds it calls
`check_and_process_emails(count=10)`, regardless of any explicit
"recipient". This is the conceptual difference from Slack/Telegram/Discord/
WhatsApp where heartbeat sends *outbound* tick output.

### 4.7 Per-channel schema modules

| File | Purpose |
|------|---------|
| `slack/schemas.py` | `SlackEventResponse` (`status: str = "ok"`), `SlackChallengeResponse` (`challenge: str`) |
| `whatsapp/schemas.py` | `WhatsAppValue` / `WhatsAppChange` / `WhatsAppEntry` / `WhatsAppWebhookPayload` (Meta envelope) |
| `telegram/schemas.py` | Full Bot API mirror — `TelegramUser`, `TelegramChat`, `TelegramPhotoSize`, `TelegramAudio`, `TelegramDocument`, `TelegramVideo`, `TelegramVoice`, `TelegramVideoNote`, `TelegramSticker`, `TelegramContact`, `TelegramLocation`, `TelegramVenue`, `TelegramPollOption`, `TelegramPoll`, `TelegramMessageEntity`, `InlineKeyboardButton`, `InlineKeyboardMarkup`, `KeyboardButton`, `ReplyKeyboardMarkup`, `ReplyKeyboardRemove`, `TelegramMessage`, `TelegramCallbackQuery`, `TelegramWebhookPayload` |
| `discord/schemas.py` | `DiscordUser`, `DiscordGuild`, `DiscordChannel`, `DiscordAttachment`, `DiscordEmbed*`, `DiscordEmoji`, `DiscordReaction`, `DiscordComponent`, `DiscordMember`, `DiscordMessageReference`, `DiscordMessage`, `DiscordInteractionData`, `DiscordInteraction`, `DiscordGatewayPayload` |
| `gmail/schemas.py` | `CheckEmailsResponse(status, processed_count, message_ids)`, `AgentEmailResponse(action: Literal["reply","ignore"], reply_body, reasoning)` |
| `mail/schemas.py` | `AttachmentInfo`, `EmailSummary` (with `from` alias `sender`), `CheckEmailsResponse`, `EmailListResponse`, `SendEmailRequest`, `SearchEmailRequest`, `MailboxStatusResponse`, `AgentEmailResponse` |

## 5. Cross-file relationships

```
                            ┌────────────────────────────────┐
                            │    InterfaceManager            │
                            │  (manager.py)                  │
                            │                                │
                            │  ┌──────────────────────────┐  │
                            │  │ FastAPI app              │  │
                            │  │  ├ middleware (CORS, ID, │  │
                            │  │  │   size, timeout, exc) │  │
                            │  │  ├ /  /health  /ws/*     │  │
                            │  │  └ <interface routers>   │  │
                            │  └──────────────────────────┘  │
                            │  WebSocketManager              │
                            │  InterfaceSettings             │
                            │  auth dep (auth.py)            │
                            └─────────────┬──────────────────┘
                                          │ contains
                                          ▼
                          ┌───────────────────────────────────┐
                          │  Interface  (base.py, ABC)        │
                          │   - id / name / agent / mode      │
                          │   - storage + _chat_sessions      │
                          │   - is_reset_command, get_chat_*  │
                          │   - attach_routes()  [abstract]   │
                          │   - health_check()                │
                          └───────────────┬───────────────────┘
                                          │ subclassed
        ┌───────────────┬─────────────────┼───────────────┬───────────────┬───────────────┐
        ▼               ▼                 ▼               ▼               ▼               ▼
  SlackInterface  WhatsAppInterface  TelegramInterface  DiscordInterface  GmailInterface  MailInterface
   (slack/...)    (whatsapp/...)     (telegram/...)     (discord/...)     (gmail/...)     (mail/...)
        │               │                 │               │               │               │
        ▼               ▼                 ▼               ▼               ▼               ▼
   SlackTools      WhatsAppTools     TelegramTools     DiscordTools     GmailTools       MailTools
   (custom_tools/) (custom_tools/)   (custom_tools/)   (custom_tools/)  (custom_tools/)  (custom_tools/)
```

Direct cross-imports inside the package:

| File | Imports |
|------|---------|
| `manager.py` | `base.Interface`, `settings.InterfaceSettings`, `websocket_manager.WebSocketManager`, `schemas.{ErrorResponse,HealthCheckResponse,WebSocketStatusResponse,WebSocketConnectionInfo}`, `auth.{get_authentication_dependency,validate_websocket_token}` |
| `base.py` | `schemas.{InterfaceMode,InterfaceResetCommand}` and lazy `upsonic.chat.Chat`, `upsonic.storage.in_memory.InMemoryStorage` |
| `auth.py` | `settings.InterfaceSettings` |
| Each channel `.py` | `interfaces.base.Interface`, `interfaces.schemas.InterfaceMode`, its own `schemas.py`, the matching `upsonic.tools.custom_tools.<channel>` (e.g. `SlackTools`, `TelegramTools`, `DiscordTools`, `GmailTools`, `MailTools`, `WhatsAppTools`) |
| All channels except Gmail | Lazy `upsonic.agent.autonomous_agent.autonomous_agent.AutonomousAgent` for heartbeat detection |
| WhatsApp | Also imports `upsonic.utils.integrations.whatsapp.{get_media_async, send_image_message_async, typing_indicator_async, upload_media_async}` |
| Telegram, Discord, Slack | Import `upsonic.tasks.tasks.Task` for TASK mode dispatch |

## 6. Public API

The canonical import paths are:

```python
from upsonic.interfaces import (
    # core
    Interface,
    InterfaceManager,
    InterfaceSettings,
    InterfaceMode,
    InterfaceResetCommand,
    # channels
    SlackInterface,    Slack,
    WhatsAppInterface, Whatsapp,
    TelegramInterface, Telegram,
    DiscordInterface,  Discord,
    GmailInterface,    Gmail,
    MailInterface,     Mail,
    # websocket
    WebSocketManager, WebSocketConnection,
    # auth
    get_authentication_dependency, validate_websocket_token,
    # response models
    HealthCheckResponse, ErrorResponse,
    WhatsAppWebhookPayload, TelegramWebhookPayload, DiscordGatewayPayload,
    WebSocketMessage, WebSocketConnectionInfo, WebSocketStatusResponse,
)
```

### 6.1 Constructor cheat sheet

```python
# Slack
SlackInterface(
    agent,
    signing_secret=None, verification_token=None,
    name="Slack", reply_to_mentions_only=True,
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_user_ids=None, stream=False, heartbeat_channel=None,
)

# WhatsApp
WhatsAppInterface(
    agent,
    verify_token=None, app_secret=None,
    name="WhatsApp",
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_numbers=None, heartbeat_recipient=None,
)

# Telegram
TelegramInterface(
    agent,
    bot_token=None, name="Telegram",
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_user_ids=None,
    webhook_secret=None, webhook_url=None,
    parse_mode="HTML", disable_web_page_preview=False,
    disable_notification=False, protect_content=False,
    reply_in_groups=True, reply_in_channels=False,
    process_edited_messages=False, process_callback_queries=True,
    typing_indicator=True, max_message_length=4096,
    stream=False, heartbeat_chat_id=None,
)

# Discord
DiscordInterface(
    agent,
    bot_token=None, name="Discord",
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_user_ids=None, allowed_channel_ids=None, allowed_guild_ids=None,
    intents=DEFAULT_INTENTS,
    typing_indicator=True, stream=False, max_message_length=2000,
    process_dm=True, process_guild_messages=True,
    heartbeat_channel_id=None,
)

# Gmail
GmailInterface(
    agent,
    name="Gmail", credentials_path=None, token_path=None, api_secret=None,
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_emails=None,
)

# Mail (generic SMTP/IMAP)
MailInterface(
    agent,
    name="Mail",
    smtp_host=None, smtp_port=None,
    imap_host=None, imap_port=None,
    username=None, password=None, use_ssl=False,
    from_address=None, api_secret=None,
    mode=InterfaceMode.TASK, reset_command="/reset", storage=None,
    allowed_emails=None, mailbox="INBOX",
)

# Manager
InterfaceManager(
    interfaces=[...], settings=InterfaceSettings(), name=None, id=None,
)
manager.serve(host="localhost", port=7777, reload=False, workers=None,
              access_log=False)
```

### 6.2 Environment variables

| Var | Used by |
|-----|---------|
| `UPSONIC_INTERFACE_SECURITY_KEY` | `InterfaceSettings.security_key` (manager-wide Bearer auth) |
| `UPSONIC_INTERFACE_*` | All other `InterfaceSettings` fields |
| `SLACK_SIGNING_SECRET`, `SLACK_VERIFICATION_TOKEN` | `SlackInterface` |
| `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_APP_SECRET` | `WhatsAppInterface` |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `TELEGRAM_WEBHOOK_URL` | `TelegramInterface` |
| `DISCORD_BOT_TOKEN` | `DiscordInterface` |
| `GMAIL_API_SECRET` | `GmailInterface` (`/gmail/check` header) |
| `MAIL_API_SECRET`, `MAIL_SMTP_HOST/PORT`, `MAIL_IMAP_HOST/PORT`, `MAIL_USERNAME`, `MAIL_PASSWORD` | `MailInterface` (consumed inside `MailTools`) |

## 7. Integration with rest of Upsonic

This package is the *outermost* layer of the framework. Inwards it touches:

| Upsonic subsystem | How `interfaces/` uses it |
|-------------------|---------------------------|
| `upsonic.agent.Agent` | TASK-mode dispatch via `agent.do_async(task, return_output=True)` and (for streaming) `agent.astream(task, events=False)`. After each run reads `agent.get_run_output()` / `output.get_last_model_response()` for text + `model_response.images` (WhatsApp) and `model_response.thinking` (Slack/WhatsApp italic preface). For HITL: reads `output.is_paused` + `output.pause_reason == "confirmation"`, mutates `output.active_requirements[i].confirm()` / `.reject()`, then `agent.continue_run_async(run_id, requirements, return_output=True)`. |
| `upsonic.agent.autonomous_agent.AutonomousAgent` | Heartbeat detection — channels lazy-import this class and only spawn the heartbeat task if `isinstance(self.agent, AutonomousAgent) and self.agent.heartbeat`. The tick uses `agent.heartbeat_period * 60` and `await self.agent.aexecute_heartbeat()`. Reset commands likewise call `agent.execute_workspace_greeting_async()` if `agent.workspace` is set, otherwise fall back to a static "Your conversation has been reset…" string. |
| `upsonic.chat.Chat` | CHAT mode. Built lazily by `Interface.get_chat_session(user_id)` with `session_id=f"{name.lower()}_{user_id}"`, `full_session_memory=True`, `summary_memory=False`, `user_analysis_memory=False`. Channels invoke via `chat.invoke(text, return_run_output=True)` (or `chat.stream(...)`) and consume the `ChatRunResult` (`.text`, `.run_output`). |
| `upsonic.storage.Storage` | Pluggable backend for chat sessions. Defaults to `InMemoryStorage` if not passed. |
| `upsonic.tasks.tasks.Task` | Used by every interface to wrap inbound user input + media: `Task(description=text, attachments=[...])` and (Gmail/Mail) `Task(..., response_format=AgentEmailResponse)`. |
| `upsonic.tools.custom_tools.<channel>` | The actual API SDK wrappers. `SlackTools`, `WhatsAppTools`, `TelegramTools`, `DiscordTools`, `GmailTools`, `MailTools` are owned by their respective interfaces and used both for sending (`asend_message`, `asend_text_message`, `asend_message_thread`, `aedit_message_text`, `acreate_interaction_response`, `atrigger_typing`, `asend_email`, `aset_webhook`, etc.) and for downloading (`aget_file`, `adownload_file`, `aget_raw_attachments`, `aget_unread_messages_raw`, `_api_request`). |
| `upsonic.utils.printing` | All channel logging goes through `info_log`, `debug_log`, `error_log`. |
| `upsonic._utils.now_utc` | Used by `schemas.py` defaults. |

Outwards, the manager is just a FastAPI app — meaning users can also do
`manager.get_app()` and embed it in their own ASGI deployment (gunicorn,
hypercorn, mangum) without using `manager.serve()`.

## 8. End-to-end flow

### 8.1 Boot

```python
from upsonic import Agent
from upsonic.interfaces import (
    InterfaceManager, InterfaceSettings,
    SlackInterface, WhatsAppInterface, TelegramInterface,
)

agent = Agent("openai/gpt-4o")

slack    = SlackInterface(agent=agent, mode="task")
whatsapp = WhatsAppInterface(agent=agent, mode="chat",
                             allowed_numbers=["905551234567"])
telegram = TelegramInterface(agent=agent, mode="chat",
                             webhook_url="https://your-host.example.com")

settings = InterfaceSettings()  # picks up UPSONIC_INTERFACE_* env vars
manager  = InterfaceManager(interfaces=[slack, whatsapp, telegram],
                            settings=settings)

manager.serve(host="0.0.0.0", port=7777)
```

What happens at construction time:

1. `InterfaceManager.__init__` → `_create_app()` builds a FastAPI app with
   middleware: request-ID, CORS (if `cors_enabled`), TrustedHost (if
   configured), upload-size + timeout enforcement, global exception
   handler.
2. `_attach_interface_routes()` calls each `interface.attach_routes()` and
   includes the returned router → `/slack/*`, `/whatsapp/*`,
   `/telegram/*` are now mounted.
3. `_add_core_routes()` mounts `/`, `/health`, `/ws/{client_id}`,
   `/ws/status`, all sharing the same `auth_dep = get_authentication_dependency(self.settings)`.
4. `manager.serve(...)` builds `uvicorn.Config(app=self.app, host=..., port=...,
   workers=..., reload=..., access_log=..., log_level=settings.log_level.lower())`
   and runs `uvicorn.Server(config).run()`.
5. FastAPI lifespan startup fires: each channel's `@router.on_event("startup")`
   handlers run — Telegram auto-installs its webhook, Discord opens the
   Gateway WS, every channel that wraps an `AutonomousAgent` starts its
   heartbeat task.

### 8.2 Inbound message — TASK mode (Slack example)

```
Slack             POST /slack/events       Manager + middleware     SlackInterface
─────             ───────────────────      ────────────────────      ──────────────
              ──> [body, X-Slack-Signature, X-Slack-Request-Timestamp]
                                          ──> request-ID, CORS,
                                              size & timeout checks
                                                                  ─> _verify_slack_signature
                                                                  ─> dedup (event_ts cache)
                                                                  ─> filter (reply_to_mentions_only,
                                                                              channel_type == im)
                                                                  ─> is_user_allowed
                                                                  ─> is_reset_command -> short-circuit
                                                                  ─> _process_event_task_mode
                                                                       └─ Task(text)
                                                                       └─ agent.do_async(task)
                                                                       └─ run_result.get_last_model_response()
                                                                       └─ _send_slack_message(channel, ts, text)
              <── 200 SlackEventResponse(status="ok")
```

(If `stream=True`, the inner block instead calls `agent.astream(task)` and
streams via `_stream_to_slack` which uses `slack_tools.update_message`
every 0.5 s.)

### 8.3 Inbound message — CHAT mode (Telegram with HITL example)

1. `POST /telegram/webhook` → `TelegramInterface.webhook` validates
   `X-Telegram-Bot-Api-Secret-Token`, parses body as
   `TelegramWebhookPayload`, schedules `_process_update(update)` as a
   `BackgroundTask`.
2. `_process_update` → `_process_message(update.message)`:
   - Filters `chat.type` against `reply_in_groups` / `reply_in_channels`.
   - Whitelist via `is_user_allowed(user.id)`.
   - Sets `_auto_heartbeat_chat_id` if not already set.
   - Reset command → `_handle_reset_command`.
   - Sends typing indicator `sendChatAction(typing)`.
   - Routes by content type → `_process_text_message` →
     `_process_chat_mode(text, user_id, chat_id, message)`.
3. `_process_chat_mode`:
   - `chat = await self.aget_chat_session(str(user_id))` — base class
     creates `Chat(session_id=f"telegram_{user_id}", ...)` on first call.
   - `result = await chat.invoke(text, return_run_output=True)`.
   - If `result.run_output.is_paused and pause_reason == "confirmation"`:
     - `_send_confirmation_and_store(run_output, chat_id, user_id,
       message_thread_id, "chat")` posts a message with two inline
       buttons (`callback_data="cfm:<key>:0:y|n"`), stores
       `{run_id, output, chat_id, user_id, message_thread_id, mode}` in
       `self._pending_confirmations[key]`.
   - Otherwise: `telegram_tools.asend_message(chat_id, text=reply_text,
     reply_to_message_id=message.message_id)`.
4. User clicks "Confirm" / "Reject" → Telegram sends
   `update.callback_query` to the same webhook → `_process_callback_query`:
   - Acknowledges via `aanswer_callback_query`.
   - Pops `state = self._pending_confirmations[key]`.
   - Mutates the agent's pending requirement: `first_req.confirm()` or
     `first_req.reject()`.
   - Calls `agent.continue_run_async(run_id, requirements,
     return_output=True)`.
   - If still paused → another confirmation message; else send the final
     reply.

### 8.4 Inbound polling — Mail / Gmail

```
Operator (cron or AutonomousAgent heartbeat tick)
   │
   ▼
POST /mail/check (header X-Upsonic-Mail-Secret)   ── or ──   internal _heartbeat_loop tick
   │
   ▼
MailInterface.check_and_process_emails(count=10)
   ├─ aget_unread_emails(count, mailbox)            (MailTools, IMAP)
   ├─ for each:
   │    ├─ dedup _is_duplicate(uid)
   │    ├─ is_email_allowed(sender)
   │    ├─ is_reset_command(body) -> _handle_reset_command + amark_email_as_read
   │    ├─ (TASK)  Task(prompt, response_format=AgentEmailResponse, attachments=[temp_paths])
   │    │          → agent.do_async(task)
   │    │          → if response.action == "reply": _send_reply(msg, response.reply_body)
   │    └─ (CHAT)  chat.invoke(prompt, attachments=[temp_paths])
   │              → if response_text: _send_reply
   ├─ amark_email_as_read(uid, mailbox)
   └─ _mark_processed(uid)  (5-minute TTL)
```

`_send_reply` calls `mail_tools.asend_reply(to, subject, body, message_id,
references)` so threading headers (`In-Reply-To`, `References`) are
preserved.

### 8.5 WebSocket session

```
client                          /ws/{client_id}                       WebSocketManager
──────                          ───────────────                       ────────────────
   ──── connect (ws upgrade) ─────────────────►
                                  Manager checks settings.is_auth_enabled()
                                  ◄── connect(ws, client_id, metadata={ping_interval,
                                       ping_timeout}, requires_auth=<bool>)
                                                                       accept + register
                                                                       authenticated_connections[ws] = not requires_auth
                                                                       send {"event":"connected", "requires_auth":...}
   ──── {"action":"authenticate", "token":"..."} ───────────────►
                                  validate_websocket_token(...)
                                                                       authenticate_websocket(ws)
                                                                       send {"event":"authenticated", ...}
   ──── {"action":"ping"} ─────────────────────►
                                                                       send {"event":"pong", "timestamp": ...}
   ──── {"action":"message", "content":"..."} ─►
                                                                       connection.send_json({"event":"message",
                                                                            "content":..., "client_id":...})
   ──── (disconnect) ──────────────────────────►
                                                                       disconnect_websocket(ws) cleans up both maps
```

The manager's `/ws/status` endpoint is Bearer-auth-gated and returns a
`WebSocketStatusResponse` listing every connection with its UUID,
human name, client ID, `connected_at`, and metadata.

### 8.6 Heartbeat (AutonomousAgent only)

For Slack, WhatsApp, Telegram, Discord, and Mail, a heartbeat loop is
started in the channel's `@router.on_event("startup")`:

```python
def _start_heartbeat(self):
    if not isinstance(self.agent, AutonomousAgent): return
    if not self.agent.heartbeat: return
    if self._heartbeat_task and not self._heartbeat_task.done(): return
    self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

async def _heartbeat_loop(self):
    period = self.agent.heartbeat_period * 60
    while True:
        await asyncio.sleep(period)
        target = self._resolve_heartbeat_<channel-key>()  # explicit OR auto-detected
        if target is None: continue
        result = await self.agent.aexecute_heartbeat()
        if result:
            await self.<channel>_tools.asend_message(<target>, result)
```

For `MailInterface` the loop instead calls
`self.check_and_process_emails(count=10)` — i.e. heartbeat = auto-poll.

For `GmailInterface` there is no heartbeat; polling is purely manual via
`POST /gmail/check`.

---

The result is a uniform "agent-as-service" pattern: regardless of which
channel a user lives on, an Upsonic agent receives a normalized text +
optional binary attachments, executes once (TASK) or within a session
(CHAT), and replies through the same channel's native send-API — with
optional streaming, HITL confirmation, whitelisting, deduplication, and
an `AutonomousAgent` heartbeat.
