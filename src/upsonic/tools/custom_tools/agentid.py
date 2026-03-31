"""AgentID identity verification toolkit for the Upsonic framework.

Provides cryptographic identity verification for AI agents using the AgentID
protocol (https://getagentid.dev). Designed for fintech workflows where agents
must prove their identity before executing regulated operations such as
payments, trading, or compliance checks.

AgentID issues ECDSA P-256 certificates per agent, exposes a public
verification API, and maintains trust scores that can drive authorization
decisions.

Environment variables:
    AGENTID_API_KEY  -- Bearer token for authenticated endpoints (register, connect).
    AGENTID_BASE_URL -- Override the default https://getagentid.dev/api/v1 endpoint.
"""

from __future__ import annotations

import json
from os import getenv
from typing import Any, Dict, List, Optional

import httpx

from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool
from upsonic.utils.printing import error_log

_DEFAULT_BASE_URL = "https://getagentid.dev/api/v1"
_TIMEOUT = 15  # seconds


class AgentIDAuthError(RuntimeError):
    """Raised when the API key is invalid, expired, or missing."""
    pass


class AgentIDRateLimitError(RuntimeError):
    """Raised when the API rate limit is exceeded."""
    pass


class AgentIDNetworkError(RuntimeError):
    """Raised when the API is unreachable."""
    pass


class AgentIDTools(ToolKit):
    """Identity verification toolkit backed by AgentID.

    Gives Upsonic agents the ability to:

    * **verify** -- check another agent's cryptographic identity before
      trusting it with sensitive data or delegating a task.
    * **register** -- obtain a new AgentID certificate so the current
      agent can prove *its own* identity to others.
    * **discover** -- search the AgentID registry for agents by
      capability or owner.
    * **connect** -- send a verified, signed message from one agent to
      another with mutual identity checks.

    Usage::

        from upsonic import Agent, Task
        from upsonic.tools.custom_tools.agentid import AgentIDTools

        agentid = AgentIDTools(api_key="aid_...")
        agent = Agent("Compliance Bot", tools=[agentid])
        task = Task("Verify agent agent_abc123 before processing payment")
        agent.run(task)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AgentIDTools class.

        Args:
            api_key: AgentID API key (Bearer token). Falls back to the
                ``AGENTID_API_KEY`` environment variable. Required for
                ``register`` and ``connect``; optional for ``verify``
                and ``discover`` (public endpoints).
            base_url: Override the default AgentID API base URL. Falls
                back to ``AGENTID_BASE_URL`` env-var, then to
                ``https://getagentid.dev/api/v1``.
            **kwargs: ToolKit params (include_tools, exclude_tools, timeout, etc.).
        """
        super().__init__(**kwargs)

        self.api_key: Optional[str] = api_key or getenv("AGENTID_API_KEY")
        self.base_url: str = (
            base_url or getenv("AGENTID_BASE_URL") or _DEFAULT_BASE_URL
        ).rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> Dict[str, str]:
        """Return Authorization header if an API key is available."""
        if not self.api_key:
            raise ValueError(
                "AgentID API key is required for this operation. "
                "Set the AGENTID_API_KEY environment variable or pass "
                "api_key= when constructing AgentIDTools."
            )
        return {"Authorization": f"Bearer {self.api_key}"}

    def _post(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        auth: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous POST to the AgentID API.

        Args:
            path: API path (e.g. ``/agents/verify``).
            body: JSON request body.
            auth: Whether to include the Authorization header.

        Returns:
            Parsed JSON response dict.
        """
        headers = self._auth_headers() if auth else {}
        try:
            resp = httpx.post(
                f"{self.base_url}{path}",
                json=body,
                headers=headers,
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AgentIDNetworkError(f"AgentID unreachable: {exc}") from exc
        data = resp.json()
        if resp.status_code == 401:
            raise AgentIDAuthError(
                "AgentID API key is invalid or expired. "
                "Check AGENTID_API_KEY and ensure it is active."
            )
        if resp.status_code == 429:
            raise AgentIDRateLimitError(
                f"AgentID rate limit exceeded: {data.get('message', 'Too many requests')}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"AgentID API error ({resp.status_code}): "
                f"{data.get('error', 'Unknown error')}"
            )
        return data

    async def _apost(
        self,
        path: str,
        body: Dict[str, Any],
        *,
        auth: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronous POST to the AgentID API.

        Args:
            path: API path (e.g. ``/agents/verify``).
            body: JSON request body.
            auth: Whether to include the Authorization header.

        Returns:
            Parsed JSON response dict.
        """
        headers = self._auth_headers() if auth else {}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.base_url}{path}",
                    json=body,
                    headers=headers,
                    timeout=_TIMEOUT,
                    follow_redirects=True,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AgentIDNetworkError(f"AgentID unreachable: {exc}") from exc
        data = resp.json()
        if resp.status_code == 401:
            raise AgentIDAuthError(
                "AgentID API key is invalid or expired. "
                "Check AGENTID_API_KEY and ensure it is active."
            )
        if resp.status_code == 429:
            raise AgentIDRateLimitError(
                f"AgentID rate limit exceeded: {data.get('message', 'Too many requests')}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"AgentID API error ({resp.status_code}): "
                f"{data.get('error', 'Unknown error')}"
            )
        return data

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous GET to the AgentID API (public endpoints).

        Args:
            path: API path.
            params: Query parameters.

        Returns:
            Parsed JSON response dict.
        """
        try:
            resp = httpx.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AgentIDNetworkError(f"AgentID unreachable: {exc}") from exc
        data = resp.json()
        if resp.status_code == 429:
            raise AgentIDRateLimitError(
                f"AgentID rate limit exceeded: {data.get('message', 'Too many requests')}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"AgentID API error ({resp.status_code}): "
                f"{data.get('error', 'Unknown error')}"
            )
        return data

    async def _aget(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Asynchronous GET to the AgentID API (public endpoints).

        Args:
            path: API path.
            params: Query parameters.

        Returns:
            Parsed JSON response dict.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}{path}",
                    params=params,
                    timeout=_TIMEOUT,
                    follow_redirects=True,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AgentIDNetworkError(f"AgentID unreachable: {exc}") from exc
        data = resp.json()
        if resp.status_code == 429:
            raise AgentIDRateLimitError(
                f"AgentID rate limit exceeded: {data.get('message', 'Too many requests')}"
            )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"AgentID API error ({resp.status_code}): "
                f"{data.get('error', 'Unknown error')}"
            )
        return data

    # ------------------------------------------------------------------
    # Tool methods -- each has sync (@tool) + async (a-prefix) pair
    # ------------------------------------------------------------------

    @tool
    def verify_agent(self, agent_id: str) -> str:
        """Verify an AI agent's cryptographic identity via AgentID.

        Use this before trusting an agent with sensitive fintech
        operations. Returns verification status, trust score,
        certificate validity, capabilities, and owner information.

        This is a public endpoint -- no API key is required.

        Args:
            agent_id: The AgentID identifier to verify (e.g. ``agent_abc123``).

        Returns:
            A JSON string with verification results including
            ``verified``, ``trust_score``, ``certificate``, and
            ``capabilities``.
        """
        try:
            result = self._post(
                "/agents/verify",
                {"agent_id": agent_id},
                auth=False,
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID verify error: {exc}")
            return json.dumps({"error": str(exc)})

    async def averify_agent(self, agent_id: str) -> str:
        """Async version of :meth:`verify_agent`."""
        try:
            result = await self._apost(
                "/agents/verify",
                {"agent_id": agent_id},
                auth=False,
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID verify error: {exc}")
            return json.dumps({"error": str(exc)})

    @tool
    def register_agent(
        self,
        name: str,
        description: str = "",
        capabilities: Optional[str] = None,
    ) -> str:
        """Register a new agent with AgentID and obtain a certificate.

        The returned certificate contains an ECDSA P-256 key pair and a
        unique ``agent_id`` that other agents can use to verify this
        agent's identity.

        Requires an AgentID API key.

        Args:
            name: Human-readable name for the agent.
            description: Short description of the agent's purpose.
            capabilities: Comma-separated capability tags
                (e.g. ``"payments,compliance,kyc"``).

        Returns:
            A JSON string with ``agent_id``, ``certificate``, and key
            material.
        """
        try:
            caps: List[str] = (
                [c.strip() for c in capabilities.split(",") if c.strip()]
                if capabilities
                else []
            )
            result = self._post(
                "/agents/register",
                {
                    "name": name,
                    "description": description,
                    "capabilities": caps,
                    "platform": "upsonic",
                },
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID register error: {exc}")
            return json.dumps({"error": str(exc)})

    async def aregister_agent(
        self,
        name: str,
        description: str = "",
        capabilities: Optional[str] = None,
    ) -> str:
        """Async version of :meth:`register_agent`."""
        try:
            caps: List[str] = (
                [c.strip() for c in capabilities.split(",") if c.strip()]
                if capabilities
                else []
            )
            result = await self._apost(
                "/agents/register",
                {
                    "name": name,
                    "description": description,
                    "capabilities": caps,
                    "platform": "upsonic",
                },
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID register error: {exc}")
            return json.dumps({"error": str(exc)})

    @tool
    def discover_agents(
        self,
        capability: str = "",
        owner: str = "",
        limit: int = 20,
    ) -> str:
        """Search the AgentID registry for other verified agents.

        Useful for finding agents that can handle a specific fintech
        capability (e.g. ``"payments"``, ``"kyc"``, ``"compliance"``).

        This is a public endpoint -- no API key is required.

        Args:
            capability: Filter by capability keyword.
            owner: Filter by owner or organisation name.
            limit: Maximum number of results (1-100).

        Returns:
            A JSON string with a list of matching agents and their
            trust scores.
        """
        try:
            params: Dict[str, Any] = {"limit": min(int(limit), 100)}
            if capability:
                params["capability"] = capability
            if owner:
                params["owner"] = owner
            result = self._get("/agents/discover", params=params)
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID discover error: {exc}")
            return json.dumps({"error": str(exc)})

    async def adiscover_agents(
        self,
        capability: str = "",
        owner: str = "",
        limit: int = 20,
    ) -> str:
        """Async version of :meth:`discover_agents`."""
        try:
            params: Dict[str, Any] = {"limit": min(int(limit), 100)}
            if capability:
                params["capability"] = capability
            if owner:
                params["owner"] = owner
            result = await self._aget("/agents/discover", params=params)
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            error_log(f"AgentID discover error: {exc}")
            return json.dumps({"error": str(exc)})

    @tool
    def connect_agents(
        self,
        from_agent: str,
        to_agent: str,
        payload: str,
        message_type: str = "request",
    ) -> str:
        """Send a verified message from one agent to another via AgentID.

        Both agents' identities are cryptographically verified before
        the message is delivered. Returns a trust check indicating
        whether both parties are verified and a recommendation for data
        exchange safety.

        Requires an AgentID API key.

        Args:
            from_agent: The ``agent_id`` of the sending agent.
            to_agent: The ``agent_id`` of the receiving agent.
            payload: JSON string with the message payload.
            message_type: Message type: ``request``, ``response``, or
                ``notification``.

        Returns:
            A JSON string with delivery status and trust check results.
        """
        try:
            payload_dict: Dict[str, Any] = json.loads(payload) if payload else {}
            result = self._post(
                "/agents/connect",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "message_type": message_type,
                    "payload": payload_dict,
                },
            )
            return json.dumps(result, indent=2, default=str)
        except json.JSONDecodeError:
            error_log("AgentID connect error: payload is not valid JSON")
            return json.dumps({"error": "payload must be a valid JSON string"})
        except Exception as exc:
            error_log(f"AgentID connect error: {exc}")
            return json.dumps({"error": str(exc)})

    async def aconnect_agents(
        self,
        from_agent: str,
        to_agent: str,
        payload: str,
        message_type: str = "request",
    ) -> str:
        """Async version of :meth:`connect_agents`."""
        try:
            payload_dict: Dict[str, Any] = json.loads(payload) if payload else {}
            result = await self._apost(
                "/agents/connect",
                {
                    "from_agent": from_agent,
                    "to_agent": to_agent,
                    "message_type": message_type,
                    "payload": payload_dict,
                },
            )
            return json.dumps(result, indent=2, default=str)
        except json.JSONDecodeError:
            error_log("AgentID connect error: payload is not valid JSON")
            return json.dumps({"error": "payload must be a valid JSON string"})
        except Exception as exc:
            error_log(f"AgentID connect error: {exc}")
            return json.dumps({"error": str(exc)})
