"""Asqav governance integration for Upsonic agents.

Signs every tool call and agent action with ML-DSA-65 (quantum-safe)
cryptographic signatures, creating a tamper-evident audit trail.

Usage::

    from upsonic import Agent, Task
    from upsonic.integrations.asqav import AsqavGovernance

    gov = AsqavGovernance(api_key="sk_...")
    agent = Agent("openai/gpt-4o", instrument=gov)
    agent.print_do("Analyze quarterly revenue data")

    # Export audit trail (raises on backend errors so you can debug)
    gov.export_audit_json()
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

from upsonic.integrations.tracing import TracingProvider

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter

_logger = logging.getLogger(__name__)


class AsqavGovernance(TracingProvider):
    """Asqav governance integration for Upsonic agents.

    Extends the standard tracing pipeline with cryptographic signing.
    Every span (tool call, agent step, LLM invocation) gets an ML-DSA-65
    signature chained to the previous action, creating a tamper-evident
    audit trail.

    Args:
        api_key: Asqav API key (``sk_...``).
            Falls back to the ``ASQAV_API_KEY`` env var. If neither is set
            the constructor raises (use a stub provider if you really want
            unsigned tracing).
        agent_name: Name for the asqav agent identity.
            Defaults to ``"upsonic-agent"``. Reused on subsequent runs;
            asqav returns the existing agent for the same name.
        endpoint: Override for the asqav API base URL.
            Falls back to the ``ASQAV_API_URL`` env var. When neither is
            set the asqav SDK keeps its own (correct) default —
            ``https://api.asqav.com/api/v1``. Don't hardcode
            ``https://api.asqav.com`` here: that strips the ``/api/v1``
            path and breaks every SDK call.
        sign_tool_calls: Sign spans that look like real tool invocations
            (per OTel GenAI semantic conventions).
        sign_llm_calls: Sign spans that look like real LLM invocations.
        sign_agent_steps: Sign generic agent / pipeline-step spans.
        include_content: Include prompt/response content in traces.
        service_name: Service name reported in traces.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        agent_name: str = "upsonic-agent",
        endpoint: Optional[str] = None,
        sign_tool_calls: bool = True,
        sign_llm_calls: bool = True,
        sign_agent_steps: bool = True,
        include_content: bool = True,
        service_name: str = "upsonic",
    ) -> None:
        # Resolve api_key from param → env var. Empty string is treated
        # the same as missing; the asqav SDK will surface a clear
        # AuthenticationError from _init_asqav() below.
        self._api_key: Optional[str] = api_key or os.environ.get("ASQAV_API_KEY") or None
        self._agent_name: str = agent_name
        # Endpoint: pass-through. ``None`` lets the asqav SDK use its own
        # correct default which already includes ``/api/v1``.
        self._endpoint: Optional[str] = endpoint or os.environ.get("ASQAV_API_URL") or None
        self._sign_tool_calls: bool = sign_tool_calls
        self._sign_llm_calls: bool = sign_llm_calls
        self._sign_agent_steps: bool = sign_agent_steps

        self._asqav: Any = None
        self._agent: Any = None
        self._session: Any = None

        # Initialize asqav BEFORE super().__init__() because the parent's
        # __init__ calls _setup() → _create_exporter(), which builds the
        # signing exporter. The exporter needs ``self._agent`` to already
        # exist; otherwise the constructor's ordering would silently
        # capture ``None`` and zero spans would ever be signed.
        self._init_asqav()

        super().__init__(
            service_name=service_name,
            include_content=include_content,
        )

    def _init_asqav(self) -> None:
        """Initialize the asqav client and create the agent identity.

        Raises:
            ImportError: when the ``asqav`` package isn't installed.
            asqav.AuthenticationError: when ``api_key`` is missing or invalid.
            asqav.APIError: when the asqav backend rejects the call (e.g.,
                wrong endpoint, missing scope, server error). These were
                previously swallowed by a bare ``except Exception: pass``
                that left users with a "tracer" that silently never signed.
        """
        try:
            import asqav
        except ImportError as exc:
            raise ImportError(
                "asqav is required for AsqavGovernance. "
                "Install it with: pip install 'upsonic[asqav]'"
            ) from exc

        # Pass base_url=None when no endpoint is given: asqav.init() ignores
        # falsy base_url and keeps its own (correct) default.
        asqav.init(api_key=self._api_key, base_url=self._endpoint)
        self._asqav = asqav
        self._agent = asqav.Agent.create(self._agent_name)
        self._session = self._agent.start_session()

    def _create_exporter(self) -> "_SpanExporter":
        """Create the signing exporter wrapping an in-memory base."""
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        # Hand the signer a *callable* into this provider rather than a
        # captured value, so it always sees the live ``self._agent``
        # regardless of when ``_init_asqav`` ran.
        return _AsqavSigningExporter(
            inner=InMemorySpanExporter(),
            get_agent=lambda: self._agent,
            sign_tool_calls=self._sign_tool_calls,
            sign_llm_calls=self._sign_llm_calls,
            sign_agent_steps=self._sign_agent_steps,
        )

    def export_audit_json(
        self,
        *,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export the audit trail as JSON.

        By default scopes the export to this provider's own agent so users
        only see actions they actually signed. Pass ``agent_id=None``
        explicitly via ``export_audit_json(agent_id=None)`` plus a
        deliberate override to fetch every agent (your API key needs the
        global ``export:read`` scope for that).

        Args:
            agent_id: Override agent_id filter; defaults to the provider's
                own ``self._agent.agent_id``.
            start_date: Optional ISO-8601 start filter.
            end_date: Optional ISO-8601 end filter.

        Returns:
            The asqav audit-trail dict.

        Raises:
            RuntimeError: if asqav was never initialized.
            asqav.APIError: surfaced from the SDK so callers can debug
                404/403/429 responses (e.g., "missing required scope").
        """
        if self._asqav is None:
            raise RuntimeError(
                "AsqavGovernance is not initialized; cannot export audit trail."
            )
        if agent_id is None and self._agent is not None:
            agent_id = self._agent.agent_id
        return self._asqav.export_audit_json(
            start_date=start_date, end_date=end_date, agent_id=agent_id
        )

    def export_audit_csv(
        self,
        *,
        agent_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Export the audit trail as CSV. See ``export_audit_json``."""
        if self._asqav is None:
            raise RuntimeError(
                "AsqavGovernance is not initialized; cannot export audit trail."
            )
        if agent_id is None and self._agent is not None:
            agent_id = self._agent.agent_id
        return self._asqav.export_audit_csv(
            start_date=start_date, end_date=end_date, agent_id=agent_id
        )

    def shutdown(self) -> None:
        """End the asqav session and shut down tracing.

        Safe to call even if ``__init__`` failed partway: only operates on
        attributes that were actually set.
        """
        agent = getattr(self, "_agent", None)
        # asqav stores the active session id as a private attribute on its
        # Agent. Accessing it here is brittle; the getattr() default keeps
        # us safe if the SDK ever renames it.
        if agent is not None and getattr(agent, "_session_id", None) is not None:
            try:
                agent.end_session()
            except Exception as exc:  # pragma: no cover - shutdown best-effort
                _logger.warning("AsqavGovernance: end_session failed: %s", exc)

        if getattr(self, "_tracer_provider", None) is not None:
            super().shutdown()


class _AsqavSigningExporter:
    """Wraps a SpanExporter to sign relevant spans via asqav before export."""

    # OTel GenAI semantic-convention attribute keys used to recognize real
    # LLM invocations (vs. internal pipeline orchestration spans).
    _LLM_ATTR_KEYS: Tuple[str, ...] = (
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.operation.name",
    )
    # Attribute keys that mark a real tool execution span.
    _TOOL_ATTR_KEYS: Tuple[str, ...] = ("gen_ai.tool.name", "gen_ai.tool.call.id")

    def __init__(
        self,
        inner: Any,
        get_agent: Callable[[], Any],
        sign_tool_calls: bool = True,
        sign_llm_calls: bool = True,
        sign_agent_steps: bool = True,
    ) -> None:
        self._inner = inner
        self._get_agent = get_agent
        self._sign_tool_calls = sign_tool_calls
        self._sign_llm_calls = sign_llm_calls
        self._sign_agent_steps = sign_agent_steps

    def export(self, spans: Any) -> Any:
        """Sign relevant spans and forward to the inner exporter."""
        agent = self._get_agent()
        if agent is not None:
            for span in spans:
                self._maybe_sign(agent, span)
        return self._inner.export(spans)

    def _classify(
        self, name: str, attrs: Dict[str, Any]
    ) -> Optional[Tuple[str, str]]:
        """Return ``(kind, action_name)`` for a span, or ``None`` to skip.

        Classification uses OTel GenAI semantic conventions (attribute
        keys), not substring matching on the span name. This avoids
        misclassifying internal Upsonic pipeline spans like
        ``pipeline.step.tool_setup`` (which is *not* a tool execution) or
        ``pipeline.step.llm_manager`` (not an LLM call).
        """
        # Real tool execution
        if any(k in attrs for k in self._TOOL_ATTR_KEYS) or any(
            str(k).startswith("gen_ai.tool.") for k in attrs
        ):
            if not self._sign_tool_calls:
                return None
            tool_name = attrs.get("gen_ai.tool.name", name)
            return "tool", str(tool_name)
        # Real LLM invocation
        if any(k in attrs for k in self._LLM_ATTR_KEYS):
            if not self._sign_llm_calls:
                return None
            return "llm", name
        # Generic agent / pipeline-step span
        if not self._sign_agent_steps:
            return None
        return "agent", name

    def _maybe_sign(self, agent: Any, span: Any) -> None:
        """Sign a span if it matches the configured span types."""
        try:
            attrs: Dict[str, Any] = dict(span.attributes or {})
            classification = self._classify(span.name or "", attrs)
            if classification is None:
                return
            kind, action_name = classification
            agent.sign(
                action_type=f"{kind}:{action_name}",
                context={k: str(v) for k, v in attrs.items()},
            )
        except Exception as exc:
            # Tracing must never break the agent — but log so operators
            # have a fighting chance to debug signing failures (instead of
            # the previous bare ``except: pass`` that silently dropped
            # every error).
            _logger.warning(
                "AsqavGovernance: failed to sign span %r: %s",
                getattr(span, "name", "?"),
                exc,
            )

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
