"""
Upsonic Merkezi Logging ve Telemetry Konfigürasyon Sistemi

Bu modül tüm Upsonic logging ve Sentry telemetry'sini tek bir yerden yönetir.
Environment variable'lar ile log seviyelerini ve telemetry'i kontrol edebilirsiniz.

Environment Variables:
    # Logging Configuration:
    UPSONIC_LOG_LEVEL: Ana log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    UPSONIC_LOG_FORMAT: Log formatı (simple, detailed, json)
    UPSONIC_LOG_FILE: Log dosyası path'i (opsiyonel)
    UPSONIC_DISABLE_LOGGING: Tüm logging'i kapat (true/false)
    UPSONIC_DISABLE_CONSOLE_LOGGING: Console logging'i kapat (user-facing apps için)

    # Sentry Telemetry Configuration (STRICTLY OPT-IN):
    # Telemetry is fully disabled unless the host explicitly sets a DSN.
    # There is no default DSN — upsonic never ships data to a third party
    # by default, and never calls sentry_sdk.init() (so the host's own
    # Sentry configuration is never replaced).
    UPSONIC_TELEMETRY: Sentry DSN (must be explicitly set; "false" or unset disables)
    UPSONIC_ENVIRONMENT: Environment name (production, development, staging)
    UPSONIC_SENTRY_SAMPLE_RATE: Traces sample rate (0.0 - 1.0, default: 0.0)
    UPSONIC_SENTRY_PROFILE_SESSION_SAMPLE_RATE: Profile sample rate (0.0 - 1.0, default: 0.0)

    # Modül bazlı seviye kontrolü:
    UPSONIC_LOG_LEVEL_LOADERS: Sadece loaders için log seviyesi
    UPSONIC_LOG_LEVEL_TEXT_SPLITTER: Sadece text_splitter için
    UPSONIC_LOG_LEVEL_VECTORDB: Sadece vectordb için
    UPSONIC_LOG_LEVEL_AGENT: Sadece agent için

Kullanım:
    from upsonic.utils.logging_config import setup_logging, enable_telemetry

    # Logging
    setup_logging(level="DEBUG", log_file="upsonic.log")

    # Telemetry (opt-in)
    enable_telemetry(dsn="https://...@sentry.io/123")
    # or set UPSONIC_TELEMETRY env var and call enable_telemetry() with no args
"""

import logging
import os
import sys
import atexit
from typing import Optional, Dict, Literal, Any, TYPE_CHECKING
from pathlib import Path
import dotenv

if TYPE_CHECKING:  # for type hints only — never executed at runtime
    import sentry_sdk

# NOTE: ``sentry_sdk`` is intentionally NOT imported at module top-level.
# Importing it here would pull the entire Sentry SDK (and its transitive
# integrations) into every worker that touches upsonic, even when telemetry
# is disabled. The module-level ``__getattr__`` below performs a lazy import
# on first attribute access, and ``enable_telemetry()`` reaches the module
# through that same channel (so existing ``patch(...sentry_sdk)`` tests keep
# working without modification).


def __getattr__(name: str) -> Any:
    """Lazy module attribute access.

    Defers ``import sentry_sdk`` until something actually reads the
    ``sentry_sdk`` attribute on this module. Result is cached in module
    globals so subsequent accesses are free.
    """
    if name == "sentry_sdk":
        import sentry_sdk as _sentry_sdk  # local import — runs once
        globals()["sentry_sdk"] = _sentry_sdk
        return _sentry_sdk
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Load environment variables from current working directory (where user runs their script)
# This ensures .env is found even when package is installed in site-packages
cwd = Path(os.getcwd())
env_path = cwd / ".env"
if env_path.exists():
    dotenv.load_dotenv(env_path, override=False)
else:
    # Fallback: search from current directory upwards (default behavior)
    dotenv.load_dotenv(override=False)

# Log level mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default log formats
LOG_FORMATS = {
    "simple": "%(levelname)-8s | %(name)-40s | %(message)s",
    "detailed": "%(asctime)s | %(levelname)-8s | %(name)-40s | %(funcName)-20s | %(message)s",
    "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
}

# Modül grupları için logger pattern'leri
MODULE_PATTERNS = {
    "loaders": "upsonic.loaders",
    "text_splitter": "upsonic.text_splitter",
    "vectordb": "upsonic.vectordb",
    "agent": "upsonic.agent",
    "team": "upsonic.team",
    "tools": "upsonic.tools",
    "cache": "upsonic.cache",
    "memory": "upsonic.memory",
    "embeddings": "upsonic.embeddings",
}

# Global flags to track configuration
_LOGGING_CONFIGURED = False
_SENTRY_CONFIGURED = False

# Isolated Sentry state for upsonic. We deliberately do NOT mutate the global
# sentry_sdk hub/scope (no sentry_sdk.init() and no Hub() construction — Hub's
# 2.x ctor mutates the global isolation scope). capture_exception() routes
# through this private Scope instead of the global one.
_upsonic_client: "Optional[sentry_sdk.Client]" = None
_upsonic_scope: "Optional[sentry_sdk.Scope]" = None


def get_env_log_level(key: str, default: str = "INFO") -> int:
    """
    Environment variable'dan log seviyesi al.

    Args:
        key: Environment variable ismi
        default: Default seviye

    Returns:
        logging.LEVEL integer değeri
    """
    level_str = os.getenv(key, default).upper()
    return LOG_LEVELS.get(level_str, logging.INFO)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Environment variable'dan boolean değer al."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_bool_optional(key: str) -> "Optional[bool]":
    """ Gets a boolean value from an environment variable, returns None if the variable is not set.
    Args:
        key: The environment variable name
    Returns:
        The boolean value of the environment variable, or None if the variable is not set
    """
    value = os.getenv(key)
    if value is None:
        return None
    value_lower = value.lower()
    return value_lower in ("true", "1", "yes", "on")


def enable_telemetry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    sample_rate: Optional[float] = None,
    profile_session_sample_rate: Optional[float] = None,
) -> bool:
    """
    Explicit opt-in for upsonic telemetry.

    Constructs an isolated ``sentry_sdk.Client`` bound to a local ``Hub``.
    This deliberately does NOT call ``sentry_sdk.init()``, so the host
    application's global Sentry configuration (DSN, integrations,
    before_send hooks, etc.) is left untouched.

    Args:
        dsn: Sentry DSN. If ``None``, falls back to ``UPSONIC_TELEMETRY``
            env var. There is no default DSN — if neither is provided,
            this is a no-op.
        environment: Environment label. Defaults to ``UPSONIC_ENVIRONMENT``
            or ``"production"``.
        sample_rate: Traces sample rate (0.0 - 1.0). Defaults to
            ``UPSONIC_SENTRY_SAMPLE_RATE`` or ``0.0``.
        profile_session_sample_rate: Profile session sample rate. Defaults
            to ``UPSONIC_SENTRY_PROFILE_SESSION_SAMPLE_RATE`` or ``0.0``.

    Returns:
        ``True`` if telemetry was enabled, ``False`` otherwise.
    """
    global _upsonic_client, _upsonic_scope  # noqa: PLW0603

    # Skip Sentry on Python 3.14+ (pydantic/fastapi compat issues)
    if sys.version_info >= (3, 14):
        return False

    if dsn is None:
        dsn = os.getenv("UPSONIC_TELEMETRY", "").strip()

    if not dsn or dsn.lower() == "false":
        return False

    if environment is None:
        environment = os.getenv("UPSONIC_ENVIRONMENT", "production")
    if sample_rate is None:
        sample_rate = float(os.getenv("UPSONIC_SENTRY_SAMPLE_RATE", "0.0"))
    if profile_session_sample_rate is None:
        profile_session_sample_rate = float(
            os.getenv("UPSONIC_SENTRY_PROFILE_SESSION_SAMPLE_RATE", "0.0")
        )

    try:
        from upsonic.utils.package.get_version import get_library_version
        release = f"upsonic@{get_library_version()}"
    except (ImportError, AttributeError, ValueError):
        release = "upsonic@unknown"

    # Resolve sentry_sdk through the module attribute so that:
    #   - the lazy ``__getattr__`` performs the actual import on demand
    #   - unittest.mock.patch('logging_config.sentry_sdk') replacements are honored
    sentry_sdk = sys.modules[__name__].sentry_sdk

    # Build an ISOLATED client. No integrations are passed, which means:
    #  * no LoggingIntegration -> we do not monkey-patch logging.Logger.callHandlers
    #    (this was the source of the Temporal sandbox _DeadlockError)
    #  * default integrations are skipped to keep the surface minimal in
    #    sandboxed/restricted-import environments
    _upsonic_client = sentry_sdk.Client(
        dsn=dsn,
        traces_sample_rate=sample_rate,
        release=release,
        server_name="upsonic_client",
        environment=environment,
        profile_session_sample_rate=profile_session_sample_rate,
        integrations=[],
        default_integrations=False,
    )
    # Bind the client to a private Scope. Unlike sentry_sdk.Hub(client) — whose
    # 2.x ctor mutates the global isolation scope — Scope().set_client() leaves
    # the global hub untouched.
    _upsonic_scope = sentry_sdk.Scope()
    _upsonic_scope.set_client(_upsonic_client)

    try:
        from upsonic.utils.package.system_id import get_system_id
        _upsonic_scope.set_user({"id": get_system_id()})
    except Exception:
        pass

    def _flush_upsonic_client() -> None:
        try:
            if _upsonic_client is not None:
                _upsonic_client.flush(timeout=2.0)
        except (RuntimeError, TimeoutError, OSError):
            pass

    atexit.register(_flush_upsonic_client)

    logger = logging.getLogger(__name__)
    logger.debug("Upsonic telemetry enabled (isolated Sentry client)")
    return True


def disable_telemetry() -> None:
    """Disable upsonic telemetry by clearing the isolated client/scope."""
    global _upsonic_client, _upsonic_scope  # noqa: PLW0603
    _upsonic_client = None
    _upsonic_scope = None


def is_telemetry_enabled() -> bool:
    """Return ``True`` if upsonic's isolated telemetry client is active."""
    return _upsonic_client is not None


def capture_exception(exc: "Optional[BaseException]" = None) -> None:
    """Capture an exception via upsonic's isolated Sentry scope.

    Silently no-ops when telemetry is disabled. Does NOT touch the global
    Sentry hub, so host applications never see upsonic's events.
    """
    if _upsonic_scope is not None:
        _upsonic_scope.capture_exception(exc)


def setup_sentry() -> None:
    """Backward-compatible entry point — strictly opt-in.

    Reads ``UPSONIC_TELEMETRY`` and, only if explicitly set to a DSN,
    enables an *isolated* Sentry client. There is no default DSN and
    ``sentry_sdk.init()`` is never called, so the host application's
    global Sentry configuration is left untouched.
    """
    global _SENTRY_CONFIGURED  # noqa: PLW0603

    if _SENTRY_CONFIGURED:
        return
    _SENTRY_CONFIGURED = True
    enable_telemetry()


def setup_logging(
    level: Optional[str] = None,
    log_format: Literal["simple", "detailed", "json"] = "simple",
    log_file: Optional[str] = None,
    force_reconfigure: bool = False,
    disable_existing_loggers: bool = False,  # noqa: ARG001
    enable_console: bool = True,
) -> None:
    """
    Upsonic logging sistemini yapılandır.

    Bu fonksiyon:
    1. Ana Upsonic logger'ını yapılandırır
    2. Modül bazlı log seviyelerini ayarlar
    3. Console ve file handler'ları ekler
    4. Rich-based printing.py ile entegre çalışır

    Args:
        level: Ana log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               None ise UPSONIC_LOG_LEVEL env var kullanılır
        log_format: Log formatı (simple, detailed, json)
        log_file: Log dosyası path'i (opsiyonel)
        force_reconfigure: True ise mevcut konfigürasyonu override et
        disable_existing_loggers: True ise diğer logger'ları kapat
        enable_console: False ise console handler ekleme (user-facing apps için)
                       Rich printing.py kullanılıyorsa False olmalı

    Examples:
        # Basit kullanım
        setup_logging(level="DEBUG")

        # Dosyaya loglama
        setup_logging(level="INFO", log_file="/var/log/upsonic.log")

        # User-facing app (console kapalı, sadece file/Sentry)
        setup_logging(level="INFO", log_file="/var/log/upsonic.log", enable_console=False)
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603

    # Eğer daha önce konfigüre edildiyse ve force değilse, skip et
    if _LOGGING_CONFIGURED and not force_reconfigure:
        return

    # Sentry'yi de initialize et (ilk kez çağrılıyorsa)
    setup_sentry()

    # Logging disabled mi kontrol et
    if get_env_bool("UPSONIC_DISABLE_LOGGING"):
        logging.getLogger("upsonic").addHandler(logging.NullHandler())
        _LOGGING_CONFIGURED = True
        return

    # Ana log seviyesini belirle (öncelik sırası: parametre > env var > default)
    if level is None:
        main_level = get_env_log_level("UPSONIC_LOG_LEVEL", "INFO")
    else:
        main_level = LOG_LEVELS.get(level.upper(), logging.INFO)

    # Log formatını al (env var'dan veya parametreden)
    format_key = os.getenv("UPSONIC_LOG_FORMAT", log_format).lower()
    log_format_str = LOG_FORMATS.get(format_key, LOG_FORMATS["simple"])

    # Log dosyasını al (env var'dan veya parametreden)
    log_file_path = os.getenv("UPSONIC_LOG_FILE", log_file)

    # Ana Upsonic logger'ını al
    upsonic_logger = logging.getLogger("upsonic")
    upsonic_logger.setLevel(main_level)
    upsonic_logger.propagate = True  # Parent logger'lara propagate et

    # Mevcut handler'ları temizle (reconfigure durumunda)
    if force_reconfigure:
        upsonic_logger.handlers.clear()

    # Formatter oluştur
    formatter = logging.Formatter(log_format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler ekle (sadece enable_console=True ise)
    # User-facing apps printing.py kullanır, console handler gereksiz
    if enable_console and not get_env_bool("UPSONIC_DISABLE_CONSOLE_LOGGING"):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(main_level)
        console_handler.setFormatter(formatter)
        upsonic_logger.addHandler(console_handler)

    # File handler ekle (eğer belirtildiyse)
    if log_file_path:
        try:
            file_path = Path(log_file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
            file_handler.setLevel(main_level)
            file_handler.setFormatter(formatter)
            upsonic_logger.addHandler(file_handler)
        except (OSError, PermissionError, ValueError) as e:
            # File handler eklenemezse sadece uyar, devam et
            upsonic_logger.warning("Could not setup file logging to %s: %s", log_file_path, e)

    # Modül bazlı log seviyelerini ayarla
    _configure_module_log_levels()

    # NullHandler ekle (eğer hiç handler yoksa)
    if not upsonic_logger.handlers:
        upsonic_logger.addHandler(logging.NullHandler())

    _LOGGING_CONFIGURED = True

    # Debug mesajı (sadece DEBUG modunda görünür)
    upsonic_logger.debug(
        "Upsonic logging configured: level=%s, format=%s",
        logging.getLevelName(main_level),
        format_key
    )


def _configure_module_log_levels() -> None:
    """
    Modül bazlı log seviyelerini environment variable'lardan ayarla.

    Environment Variables:
        UPSONIC_LOG_LEVEL_LOADERS: upsonic.loaders için seviye
        UPSONIC_LOG_LEVEL_TEXT_SPLITTER: upsonic.text_splitter için seviye
        etc.
    """
    for module_key, module_pattern in MODULE_PATTERNS.items():
        env_key = f"UPSONIC_LOG_LEVEL_{module_key.upper()}"
        env_value = os.getenv(env_key)

        if env_value:
            level = LOG_LEVELS.get(env_value.upper())
            if level:
                module_logger = logging.getLogger(module_pattern)
                module_logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Upsonic için logger al.

    Bu fonksiyon kullanılması önerilir, çünkü:
    1. Logging ilk kez kullanılırken otomatik konfigüre eder
    2. Modül ismini normalize eder

    Args:
        name: Logger ismi (genelde __name__)

    Returns:
        Configured logger instance

    Example:
        # Modül başında
        from upsonic.utils.logging_config import get_logger
        logger = get_logger(__name__)

        # Kullanım
        logger.debug("Debug mesajı")
        logger.info("Info mesajı")
    """
    # İlk kez kullanılıyorsa otomatik konfigüre et
    if not _LOGGING_CONFIGURED:
        setup_logging()

    return logging.getLogger(name)


def set_module_log_level(module: str, level: str) -> None:
    """
    Belirli bir modül için log seviyesini runtime'da değiştir.

    Args:
        module: Modül pattern'i (örn: "loaders", "text_splitter")
                veya tam logger ismi (örn: "upsonic.loaders.base")
        level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        # Sadece loaders'ı WARNING'e çek
        set_module_log_level("loaders", "WARNING")

        # Spesifik bir modül
        set_module_log_level("upsonic.text_splitter.agentic", "DEBUG")
    """
    log_level = LOG_LEVELS.get(level.upper())
    if not log_level:
        raise ValueError(f"Invalid log level: {level}")

    # Eğer kısa isim kullanıldıysa (örn: "loaders"), pattern'e çevir
    logger_name = MODULE_PATTERNS.get(module, module)

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)


def disable_logging() -> None:
    """Tüm Upsonic logging'ini kapat."""
    upsonic_logger = logging.getLogger("upsonic")
    upsonic_logger.handlers.clear()
    upsonic_logger.addHandler(logging.NullHandler())
    upsonic_logger.setLevel(logging.CRITICAL + 1)  # Hiçbir şey loglanmasın


def get_current_log_levels() -> Dict[str, str]:
    """
    Tüm Upsonic logger'larının mevcut seviyelerini göster.

    Returns:
        Logger ismi -> log seviyesi mapping'i

    Example:
        >>> from upsonic.utils.logging_config import get_current_log_levels
        >>> levels = get_current_log_levels()
        >>> print(levels)
        {
            'upsonic': 'INFO',
            'upsonic.loaders': 'WARNING',
            'upsonic.text_splitter': 'DEBUG',
            ...
        }
    """
    levels = {}

    # Ana logger
    upsonic_logger = logging.getLogger("upsonic")
    levels["upsonic"] = logging.getLevelName(upsonic_logger.level)

    # Modül logger'ları
    for _module_key, module_pattern in MODULE_PATTERNS.items():
        logger = logging.getLogger(module_pattern)
        if logger.level != logging.NOTSET:  # Sadece explicitly set edilmişleri göster
            levels[module_pattern] = logging.getLevelName(logger.level)

    return levels


def memory_debug_log(memory_debug: bool, msg: str, data: Any = None) -> None:
    """Print debug log for memory operations if debug is enabled.
    
    Args:
        memory_debug: Whether memory debugging is enabled
        msg: Debug message to print
        data: Optional data to display (list or other object)
    """
    if memory_debug:
        print(f"  🔍 [MEMORY DEBUG] {msg}")
        if data is not None:
            if isinstance(data, list):
                print(f"      Count: {len(data)} items")
                for i, item in enumerate(data[:5]):
                    item_str = str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                    print(f"      [{i}] {item_str}")
                if len(data) > 5:
                    print(f"      ... and {len(data) - 5} more")
            else:
                data_str = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                print(f"      {data_str}")


_OTEL_CONFIGURED: bool = False


def setup_opentelemetry() -> None:
    """Configure OpenTelemetry tracing for Upsonic.

    Delegates all OTel bootstrapping to :class:`DefaultTracingProvider` which
    reads ``UPSONIC_OTEL_*`` environment variables (endpoint, service name,
    headers, sample rate).

    When ``UPSONIC_OTEL_ENABLED=True`` is set, automatically calls
    ``Agent.instrument_all()`` so every Agent instance is instrumented.

    This function is safe to call multiple times; subsequent calls are no-ops.
    """
    global _OTEL_CONFIGURED  # noqa: PLW0603

    if _OTEL_CONFIGURED:
        return

    try:
        from upsonic.integrations.tracing import DefaultTracingProvider

        provider: DefaultTracingProvider = DefaultTracingProvider()
        _OTEL_CONFIGURED = True

        logger = logging.getLogger(__name__)
        logger.debug(
            "OpenTelemetry configured via DefaultTracingProvider: endpoint=%s, service=%s",
            provider._endpoint,
            provider._service_name,
        )
        from upsonic.agent.agent import Agent
        Agent.instrument_all(instrument=provider)

    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning(
            "OpenTelemetry SDK packages not installed. "
            "Install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp"
        )
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.warning("Failed to configure OpenTelemetry: %s", exc)


# NOTE: Telemetry is NOT initialized at import time. It is strictly opt-in.
# Hosts that want it must either set UPSONIC_TELEMETRY=<dsn> and call
# setup_sentry()/enable_telemetry(), or call enable_telemetry(dsn=...) directly.

# Logging sadece env var varsa otomatik konfigüre edilir
if os.getenv("UPSONIC_LOG_LEVEL") or os.getenv("UPSONIC_LOG_FILE"):
    setup_logging()
else:
    # Env var yoksa sadece NullHandler ekle (library best practice)
    logging.getLogger("upsonic").addHandler(logging.NullHandler())

# Auto-configure OpenTelemetry if env vars are set
if os.getenv("UPSONIC_OTEL_ENABLED"):
    setup_opentelemetry()
