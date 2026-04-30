"""
TECH-1428: Tests for opt-in telemetry behavior.

The previous behavior had three problems:
  1. `setup_sentry()` ran at import time, calling `sentry_sdk.init()` and
     replacing the host application's global Sentry client.
  2. The DSN defaulted to upsonic's own Sentry project, so any host that
     didn't explicitly opt out shipped error logs to a third party.
  3. `LoggingIntegration` was registered globally, monkey-patching
     `logging.Logger.callHandlers` and breaking sandboxed environments
     such as Temporal workflows.

Fix surface:
  * Importing `upsonic.utils.logging_config` MUST NOT call `sentry_sdk.init`.
  * `setup_sentry()` is now a no-op when `UPSONIC_TELEMETRY` is unset; there
    is no default DSN.
  * `enable_telemetry(dsn=...)` is the explicit opt-in entry point and
    constructs an isolated `sentry_sdk.Client` bound to a local `Hub` —
    it never calls `sentry_sdk.init()`.
  * No `LoggingIntegration` is registered.
  * `capture_exception()` routes through the isolated hub, never the global
    one.
"""

import importlib
import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


def _reset_logging_config_state() -> None:
    """Reset module-level flags so each test starts from a clean slate."""
    from upsonic.utils import logging_config

    logging_config._SENTRY_CONFIGURED = False
    logging_config._upsonic_client = None
    logging_config._upsonic_scope = None


class TestNoImportTimeSideEffects(unittest.TestCase):
    """Importing the module must not touch the global Sentry SDK.

    These tests run in subprocesses so the fresh-import behavior can be
    observed without polluting sys.modules for sibling test files (which
    would leave their top-level ``from upsonic.utils.logging_config
    import ...`` references bound to an orphan module).
    """

    def _run_in_subprocess(self, script: str) -> "subprocess.CompletedProcess":
        import subprocess

        env = os.environ.copy()
        env.pop("UPSONIC_TELEMETRY", None)
        # Bypass the repo's .env, which sets UPSONIC_TELEMETRY=false and
        # would mask the bug we are reproducing.
        env["DOTENV_DISABLE"] = "1"
        return subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_importing_module_does_not_call_sentry_init(self) -> None:
        """Re-importing logging_config must not invoke sentry_sdk.init().

        The original bug shipped a hard-coded default DSN, so even a host
        that never set UPSONIC_TELEMETRY had its global Sentry client
        silently replaced.
        """
        result = self._run_in_subprocess(
            "import sys; "
            "from unittest.mock import patch; "
            # Stub load_dotenv so the repo's .env cannot reintroduce
            # UPSONIC_TELEMETRY=false (which would mask the bug).
            "patch('dotenv.load_dotenv').start(); "
            "init_calls = []; "
            "import sentry_sdk; "
            "sentry_sdk.init = lambda *a, **kw: init_calls.append((a, kw)); "
            "import upsonic.utils.logging_config; "
            "print(f'INIT_CALLS={len(init_calls)}')"
        )
        self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
        self.assertIn("INIT_CALLS=0", result.stdout)

    def test_importing_module_does_not_register_logging_integration(self) -> None:
        """No LoggingIntegration may be installed at import time."""
        result = self._run_in_subprocess(
            "import sys; "
            "from unittest.mock import patch; "
            "patch('dotenv.load_dotenv').start(); "
            "calls = []; "
            "import sentry_sdk.integrations.logging as li; "
            "_orig = li.LoggingIntegration; "
            "li.LoggingIntegration = lambda *a, **kw: calls.append((a, kw)) or _orig(*a, **kw); "
            "import upsonic.utils.logging_config; "
            "print(f'INTEGRATION_CALLS={len(calls)}')"
        )
        self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")
        self.assertIn("INTEGRATION_CALLS=0", result.stdout)


class TestSentrySdkLazyImport(unittest.TestCase):
    """sentry_sdk must not be loaded into sys.modules until first use.

    Even when telemetry is disabled, the previous code paid the cost of
    importing the entire Sentry SDK (and its transitive integrations) in
    every worker. The lazy ``__getattr__`` defers that cost until something
    actually reads ``logging_config.sentry_sdk``.

    Each test runs in a fresh subprocess so reimport state pollution can
    never leak into other tests in this file.
    """

    def _run_in_subprocess(self, script: str) -> "subprocess.CompletedProcess":
        import subprocess

        env = os.environ.copy()
        env.pop("UPSONIC_TELEMETRY", None)
        env.pop("UPSONIC_LOG_LEVEL", None)
        env.pop("UPSONIC_LOG_FILE", None)
        return subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_fresh_import_of_logging_config_does_not_load_sentry_sdk(self) -> None:
        """Importing logging_config must not pull sentry_sdk into sys.modules."""
        result = self._run_in_subprocess(
            "import sys; "
            "import upsonic.utils.logging_config; "
            "print('LOADED' if 'sentry_sdk' in sys.modules else 'NOT_LOADED')"
        )
        self.assertEqual(
            result.returncode, 0, msg=f"stderr: {result.stderr}"
        )
        self.assertIn("NOT_LOADED", result.stdout)

    def test_attribute_access_triggers_lazy_load(self) -> None:
        """Reading logging_config.sentry_sdk lazily imports the SDK."""
        result = self._run_in_subprocess(
            "import sys; "
            "from upsonic.utils import logging_config; "
            "before = 'sentry_sdk' in sys.modules; "
            "_ = logging_config.sentry_sdk; "
            "after = 'sentry_sdk' in sys.modules; "
            "print(f'BEFORE={before} AFTER={after}')"
        )
        self.assertEqual(
            result.returncode, 0, msg=f"stderr: {result.stderr}"
        )
        self.assertIn("BEFORE=False AFTER=True", result.stdout)


class TestNoDefaultDSN(unittest.TestCase):
    """The library must not ship a hard-coded fallback DSN."""

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_setup_sentry_noop_without_env_var(self) -> None:
        """setup_sentry() is a no-op when UPSONIC_TELEMETRY is unset."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client:
            logging_config.setup_sentry()
            mock_init.assert_not_called()
            mock_client.assert_not_called()

    def test_setup_sentry_noop_when_telemetry_false(self) -> None:
        """Explicit `false` keeps telemetry disabled."""
        os.environ["UPSONIC_TELEMETRY"] = "false"
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client:
            logging_config.setup_sentry()
            mock_init.assert_not_called()
            mock_client.assert_not_called()

    def test_module_source_contains_no_hardcoded_upsonic_dsn(self) -> None:
        """No literal upsonic ingest DSN may remain in the source."""
        import upsonic.utils.logging_config as mod

        source = open(mod.__file__, "r", encoding="utf-8").read()
        self.assertNotIn("ingest.us.sentry.io", source)
        self.assertNotIn("o4508336623583232", source)


class TestIsolatedClientNotGlobalInit(unittest.TestCase):
    """enable_telemetry() must use an isolated Client + Hub, never global init."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_enable_telemetry_function_exists(self) -> None:
        """The public opt-in API must be exported."""
        from upsonic.utils import logging_config

        self.assertTrue(hasattr(logging_config, "enable_telemetry"))
        self.assertTrue(callable(logging_config.enable_telemetry))

    def test_enable_telemetry_does_not_call_sentry_init(self) -> None:
        """Even when enabled, sentry_sdk.init() must never be called."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.init") as mock_init, patch(
            "sentry_sdk.Client"
        ) as mock_client, patch("sentry_sdk.Scope"):
            mock_client.return_value = MagicMock()
            result = logging_config.enable_telemetry(dsn=self._DSN)
            self.assertTrue(result)
            mock_init.assert_not_called()
            mock_client.assert_called_once()

    def test_enable_telemetry_does_not_construct_hub(self) -> None:
        """REGRESSION GUARD — do not delete this test.

        sentry_sdk.Hub must NOT be used — its 2.x ctor mutates global state.
        Reading sentry_sdk 2.48 source: ``Hub.__init__`` calls
        ``get_global_scope().set_client(client)``, which is exactly the
        global mutation we are trying to avoid. The fix uses a bare Scope
        instead. If a future refactor reintroduces ``sentry_sdk.Hub(...)``,
        this test will fail loudly.
        """
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Hub"
        ) as mock_hub, patch("sentry_sdk.Scope", return_value=MagicMock()):
            logging_config.enable_telemetry(dsn=self._DSN)
            mock_hub.assert_not_called()

    def test_enable_telemetry_constructs_isolated_scope(self) -> None:
        """A bare Scope bound to the isolated Client must be created."""
        from upsonic.utils import logging_config

        fake_client = MagicMock(name="Client")
        fake_scope = MagicMock(name="Scope")
        with patch("sentry_sdk.Client", return_value=fake_client), patch(
            "sentry_sdk.Scope", return_value=fake_scope
        ) as mock_scope_ctor, patch("sentry_sdk.init") as mock_init:
            logging_config.enable_telemetry(dsn=self._DSN)
            mock_init.assert_not_called()
            mock_scope_ctor.assert_called_once_with()
            fake_scope.set_client.assert_called_once_with(fake_client)
            self.assertIs(logging_config._upsonic_client, fake_client)
            self.assertIs(logging_config._upsonic_scope, fake_scope)

    def test_enable_telemetry_does_not_install_logging_integration(self) -> None:
        """LoggingIntegration must never be passed to the isolated Client."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client, patch("sentry_sdk.Scope"):
            logging_config.enable_telemetry(dsn=self._DSN)
            kwargs = mock_client.call_args.kwargs
            integrations = kwargs.get("integrations") or []
            for integration in integrations:
                name = type(integration).__name__
                self.assertNotEqual(name, "LoggingIntegration")

    def test_enable_telemetry_returns_false_without_dsn(self) -> None:
        """enable_telemetry() with no DSN and no env var is a no-op."""
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client:
            result = logging_config.enable_telemetry()
            self.assertFalse(result)
            mock_client.assert_not_called()

    def test_enable_telemetry_reads_env_when_dsn_omitted(self) -> None:
        """If no DSN argument is given, fall back to UPSONIC_TELEMETRY env var."""
        os.environ["UPSONIC_TELEMETRY"] = self._DSN
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client") as mock_client, patch("sentry_sdk.Scope"):
            result = logging_config.enable_telemetry()
            self.assertTrue(result)
            self.assertEqual(mock_client.call_args.kwargs["dsn"], self._DSN)


class TestHostSentryNotHijacked(unittest.TestCase):
    """If the host already configured Sentry, upsonic must not overwrite it."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_global_hub_client_unchanged_after_enable(self) -> None:
        """The global Hub's client must be untouched after enable_telemetry().

        We snapshot the current global hub's client identity before and
        after the call. enable_telemetry() must build an isolated Client +
        Hub pair without ever swapping the global hub's bound client.
        """
        import sentry_sdk
        from upsonic.utils import logging_config

        before = sentry_sdk.Hub.current.client
        logging_config.enable_telemetry(dsn=self._DSN)
        after = sentry_sdk.Hub.current.client
        self.assertIs(before, after)


class TestCaptureExceptionUsesIsolatedHub(unittest.TestCase):
    """capture_exception() must route through the isolated hub, not the global one."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.startswith("UPSONIC_"):
                del os.environ[key]
        _reset_logging_config_state()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self.original_env)
        _reset_logging_config_state()

    def test_capture_exception_noop_when_disabled(self) -> None:
        """Without enable_telemetry(), capture_exception is a silent no-op."""
        from upsonic.utils import logging_config

        # Should not raise even though no hub is configured.
        logging_config.capture_exception(RuntimeError("boom"))

    def test_capture_exception_uses_isolated_scope(self) -> None:
        """Exceptions captured through upsonic's helper hit the isolated scope."""
        from upsonic.utils import logging_config

        fake_scope = MagicMock(name="IsolatedScope")
        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Scope", return_value=fake_scope
        ):
            logging_config.enable_telemetry(dsn=self._DSN)

        err = RuntimeError("boom")
        logging_config.capture_exception(err)
        fake_scope.capture_exception.assert_called_once_with(err)


class TestDisableTelemetry(unittest.TestCase):
    """disable_telemetry() should reset the isolated client/hub."""

    _DSN = "https://abc@example.ingest.sentry.io/1"

    def setUp(self) -> None:
        _reset_logging_config_state()

    def tearDown(self) -> None:
        _reset_logging_config_state()

    def test_disable_telemetry_clears_state(self) -> None:
        from upsonic.utils import logging_config

        with patch("sentry_sdk.Client", return_value=MagicMock()), patch(
            "sentry_sdk.Scope", return_value=MagicMock()
        ):
            logging_config.enable_telemetry(dsn=self._DSN)

        self.assertTrue(logging_config.is_telemetry_enabled())
        logging_config.disable_telemetry()
        self.assertFalse(logging_config.is_telemetry_enabled())
        self.assertIsNone(logging_config._upsonic_client)
        self.assertIsNone(logging_config._upsonic_scope)


if __name__ == "__main__":
    unittest.main()
