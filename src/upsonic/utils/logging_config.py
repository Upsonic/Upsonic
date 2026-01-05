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

    # Sentry Telemetry Configuration:
    UPSONIC_TELEMETRY: Sentry DSN (ya da "false" to disable)
    UPSONIC_ENVIRONMENT: Environment name (production, development, staging)
    UPSONIC_SENTRY_SAMPLE_RATE: Traces sample rate (0.0 - 1.0, default: 1.0)

    # Modül bazlı seviye kontrolü:
    UPSONIC_LOG_LEVEL_LOADERS: Sadece loaders için log seviyesi
    UPSONIC_LOG_LEVEL_TEXT_SPLITTER: Sadece text_splitter için
    UPSONIC_LOG_LEVEL_VECTORDB: Sadece vectordb için
    UPSONIC_LOG_LEVEL_AGENT: Sadece agent için

Kullanım:
    # Otomatik konfigürasyon (import ederken çalışır)
    from upsonic.utils.logging_config import setup_logging, sentry_sdk

    # Ya da manuel
    setup_logging(level="DEBUG", log_file="upsonic.log")

    # Sentry tracing kullanımı
    with sentry_sdk.start_transaction(op="task", name="My Task"):
        # your code here
        pass

    # Environment variable ile
    export UPSONIC_LOG_LEVEL=DEBUG
    export UPSONIC_TELEMETRY="your-sentry-dsn"
    export UPSONIC_ENVIRONMENT="production"
"""

import logging
import os
import sys
import atexit
from typing import Optional, Dict, Literal, Any
from pathlib import Path
from dotenv import load_dotenv

# Sentry SDK imports
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Load environment variables
load_dotenv()

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


def setup_sentry() -> None:
    """
    Sentry telemetry sistemini yapılandır.

    Environment Variables:
        UPSONIC_TELEMETRY: Sentry DSN URL'i veya "false" to disable
        UPSONIC_ENVIRONMENT: Environment adı (production, development, staging)
        UPSONIC_SENTRY_SAMPLE_RATE: Traces sample rate (0.0 - 1.0)

    Bu fonksiyon:
    1. Sentry SDK'yı initialize eder
    2. User ID tracking ayarlar
    3. Release bilgisi ekler
    4. Logging integration'ı aktif eder (WARNING+ loglar Sentry'e gider)
    """
    global _SENTRY_CONFIGURED  # noqa: PLW0603

    # Eğer daha önce konfigüre edildiyse, skip et
    if _SENTRY_CONFIGURED:
        return

    # Get configuration from environment
    the_dsn = os.getenv(
        "UPSONIC_TELEMETRY",
        "https://f9b529d9b67a30fae4d5b6462256ee9e@o4508336623583232.ingest.us.sentry.io/4510211809542144"
    )
    the_environment = os.getenv("UPSONIC_ENVIRONMENT", "production")
    the_sample_rate = float(os.getenv("UPSONIC_SENTRY_SAMPLE_RATE", "1.0"))
    the_profile_session_sample_rate = float(os.getenv("UPSONIC_SENTRY_PROFILE_SESSION_SAMPLE_RATE", "1.0"))

    # "false" değeri varsa Sentry'yi devre dışı bırak
    if the_dsn.lower() == "false":
        the_dsn = ""

    # Get version for release tag
    try:
        from upsonic.utils.package.get_version import get_library_version
        the_release = f"upsonic@{get_library_version()}"
    except (ImportError, AttributeError, ValueError):
        the_release = "upsonic@unknown"

    # Initialize Sentry SDK
    sentry_sdk.init(
        dsn=the_dsn,
        traces_sample_rate=the_sample_rate,
        release=the_release,
        server_name="upsonic_client",
        environment=the_environment,
        # Logging integration - INFO+ logları Sentry event olarak gönder
        # Breadcrumb için tüm level'ları yakala, event için INFO+
        integrations=[
            LoggingIntegration(
                level=logging.INFO,  # INFO+ logları Sentry'e gönder
                event_level=logging.INFO,  # INFO ve üzeri Sentry event olarak gönder
            ),
        ],
        profile_session_sample_rate=the_profile_session_sample_rate,
    )

    # Set user ID for tracking
    try:
        from upsonic.utils.package.system_id import get_system_id
        sentry_sdk.set_user({"id": get_system_id()})
    except Exception:
        pass  # System ID alınamazsa skip et

    _SENTRY_CONFIGURED = True

    # Register atexit handler to flush pending events on program exit
    # Bu sayede script/CLI kullanımında pending event'ler kaybolmaz
    if the_dsn:
        def _flush_sentry():
            """Flush pending Sentry events before program exit."""
            try:
                sentry_sdk.flush(timeout=2.0)
            except (RuntimeError, TimeoutError, OSError):
                pass  # Silent failure on exit, don't block program termination

        atexit.register(_flush_sentry)

    # Log initialization (sadece DSN varsa)
    if the_dsn:
        logger = logging.getLogger(__name__)
        logger.debug("Sentry initialized for Upsonic")


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


# Library import edildiğinde otomatik konfigüre et
# Sentry her zaman initialize edilir (DSN kontrolü setup_sentry içinde)
setup_sentry()

# Logging sadece env var varsa otomatik konfigüre edilir
if os.getenv("UPSONIC_LOG_LEVEL") or os.getenv("UPSONIC_LOG_FILE"):
    setup_logging()
else:
    # Env var yoksa sadece NullHandler ekle (library best practice)
    logging.getLogger("upsonic").addHandler(logging.NullHandler())
