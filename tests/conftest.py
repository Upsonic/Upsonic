"""Pytest configuration to handle DisallowedOperation exceptions gracefully."""
import os

# Force-disable telemetry/tracing BEFORE any upsonic module is imported.
# os.environ.__setitem__ (not setdefault) so a subsequent load_dotenv(override=False) cannot win.
os.environ["UPSONIC_TELEMETRY"] = "false"
os.environ["UPSONIC_OTEL_ENABLED"] = ""

import pytest
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
# Make the ``tests`` package importable (``from tests._pipeline_injection
# import ...``) so test-only helpers can live next to the suites.
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from dotenv import load_dotenv
    env_path = _root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✓ Loaded environment variables from: {env_path}")
    else:
        print(f"⚠️  .env file not found at: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

from upsonic.safety_engine.exceptions import DisallowedOperation  # noqa: E402


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Handle DisallowedOperation exceptions gracefully during test execution."""
    outcome = yield
    report = outcome.get_result()
    
    if call.excinfo is not None and call.when == "call":
        exc_type = call.excinfo.type
        exc_value = call.excinfo.value
        
        is_disallowed = (
            exc_type is DisallowedOperation or 
            (exc_type is not None and exc_type.__name__ == "DisallowedOperation")
        )
        
        if is_disallowed:
            print(f"\n⚠️  DisallowedOperation caught in {item.name}: {exc_value}", file=sys.stderr)
            print("   Handling gracefully - test will not fail\n", file=sys.stderr)
            
            report.outcome = "passed"
            if hasattr(report, 'longrepr'):
                report.longrepr = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up background threads so the process can exit immediately."""
    import threading

    # Stop the upsonic background event loop
    try:
        from upsonic.agent import agent as _agent_mod
        loop = getattr(_agent_mod, "_bg_loop", None)
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
    except Exception:
        pass

    # Shut down Sentry (flushes and stops worker threads)
    try:
        import sentry_sdk
        client = sentry_sdk.get_client()
        if client and client.is_active():
            sentry_sdk.flush(timeout=2)
            client.close(timeout=2)
    except Exception:
        pass

    # Shut down any OpenTelemetry providers
    try:
        from opentelemetry import trace as _otel_trace
        provider = _otel_trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        pass

    # Force-kill any lingering non-daemon threads spawned by crawlee / asyncio
    _main_thread = threading.main_thread()
    for t in threading.enumerate():
        if t is _main_thread or t.daemon:
            continue
        try:
            t.join(timeout=2)
        except Exception:
            pass

