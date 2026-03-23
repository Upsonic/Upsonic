"""Unit tests for the AgentID identity verification toolkit."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from upsonic.tools.custom_tools.agentid import AgentIDTools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agentid_tools():
    """Return an AgentIDTools instance with an explicit API key."""
    return AgentIDTools(api_key="test_key_123", base_url="https://test.getagentid.dev/api/v1")


@pytest.fixture
def agentid_tools_no_key():
    """Return an AgentIDTools instance without an API key."""
    return AgentIDTools(base_url="https://test.getagentid.dev/api/v1")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestAgentIDToolsInit:
    """Tests for AgentIDTools initialization."""

    def test_init_with_explicit_key(self) -> None:
        tools = AgentIDTools(api_key="my_key")
        assert tools.api_key == "my_key"
        assert tools.base_url == "https://getagentid.dev/api/v1"

    def test_init_with_env_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTID_API_KEY", "env_key")
        tools = AgentIDTools()
        assert tools.api_key == "env_key"

    def test_init_with_custom_base_url(self) -> None:
        tools = AgentIDTools(base_url="https://custom.example.com/api/v1/")
        assert tools.base_url == "https://custom.example.com/api/v1"

    def test_init_with_env_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENTID_BASE_URL", "https://env.example.com/api/v1")
        tools = AgentIDTools()
        assert tools.base_url == "https://env.example.com/api/v1"

    def test_init_is_toolkit(self) -> None:
        from upsonic.tools.base import ToolKit
        tools = AgentIDTools(api_key="k")
        assert isinstance(tools, ToolKit)


# ---------------------------------------------------------------------------
# verify_agent
# ---------------------------------------------------------------------------

class TestVerifyAgent:
    """Tests for the verify_agent tool method."""

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_verify_agent_success(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "verified": True,
            "agent_id": "agent_abc123",
            "trust_score": 0.95,
        }
        mock_post.return_value = mock_resp

        result = agentid_tools.verify_agent("agent_abc123")
        data = json.loads(result)

        assert data["verified"] is True
        assert data["trust_score"] == 0.95
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"] == {"agent_id": "agent_abc123"}
        # Verify endpoint does not send auth headers
        assert call_kwargs.kwargs["headers"] == {}

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_verify_agent_not_found(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "Agent not found"}
        mock_post.return_value = mock_resp

        result = agentid_tools.verify_agent("agent_nonexistent")
        data = json.loads(result)

        assert "error" in data

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_verify_agent_network_error(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_post.side_effect = Exception("Connection refused")

        result = agentid_tools.verify_agent("agent_abc123")
        data = json.loads(result)

        assert "error" in data
        assert "Connection refused" in data["error"]


# ---------------------------------------------------------------------------
# verify_agent async
# ---------------------------------------------------------------------------

class TestVerifyAgentAsync:
    """Tests for the async averify_agent method."""

    @pytest.mark.asyncio
    async def test_averify_agent_success(self, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "verified": True,
            "agent_id": "agent_abc123",
            "trust_score": 0.92,
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("upsonic.tools.custom_tools.agentid.httpx.AsyncClient", return_value=mock_client):
            result = await agentid_tools.averify_agent("agent_abc123")

        data = json.loads(result)
        assert data["verified"] is True
        assert data["trust_score"] == 0.92


# ---------------------------------------------------------------------------
# register_agent
# ---------------------------------------------------------------------------

class TestRegisterAgent:
    """Tests for the register_agent tool method."""

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_register_agent_success(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "agent_id": "agent_new456",
            "certificate": {"public_key": "..."},
        }
        mock_post.return_value = mock_resp

        result = agentid_tools.register_agent(
            name="Payment Bot",
            description="Handles payments",
            capabilities="payments,compliance",
        )
        data = json.loads(result)

        assert data["agent_id"] == "agent_new456"
        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["name"] == "Payment Bot"
        assert body["platform"] == "upsonic"
        assert body["capabilities"] == ["payments", "compliance"]
        # Auth header should be present
        assert "Authorization" in call_kwargs.kwargs["headers"]

    def test_register_agent_no_api_key(self, agentid_tools_no_key: AgentIDTools) -> None:
        result = agentid_tools_no_key.register_agent(name="Test")
        data = json.loads(result)
        assert "error" in data

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_register_agent_empty_capabilities(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"agent_id": "agent_x"}
        mock_post.return_value = mock_resp

        agentid_tools.register_agent(name="Bot")
        body = mock_post.call_args.kwargs["json"]
        assert body["capabilities"] == []


# ---------------------------------------------------------------------------
# discover_agents
# ---------------------------------------------------------------------------

class TestDiscoverAgents:
    """Tests for the discover_agents tool method."""

    @patch("upsonic.tools.custom_tools.agentid.httpx.get")
    def test_discover_agents_success(self, mock_get: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "agents": [
                {"agent_id": "agent_a", "name": "KYC Bot", "trust_score": 0.9},
            ]
        }
        mock_get.return_value = mock_resp

        result = agentid_tools.discover_agents(capability="kyc")
        data = json.loads(result)

        assert len(data["agents"]) == 1
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["capability"] == "kyc"

    @patch("upsonic.tools.custom_tools.agentid.httpx.get")
    def test_discover_agents_limit_capped(self, mock_get: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"agents": []}
        mock_get.return_value = mock_resp

        agentid_tools.discover_agents(limit=200)
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["limit"] == 100


# ---------------------------------------------------------------------------
# connect_agents
# ---------------------------------------------------------------------------

class TestConnectAgents:
    """Tests for the connect_agents tool method."""

    @patch("upsonic.tools.custom_tools.agentid.httpx.post")
    def test_connect_agents_success(self, mock_post: MagicMock, agentid_tools: AgentIDTools) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "delivered": True,
            "trust_check": {"from_verified": True, "to_verified": True},
        }
        mock_post.return_value = mock_resp

        result = agentid_tools.connect_agents(
            from_agent="agent_a",
            to_agent="agent_b",
            payload='{"action": "transfer", "amount": 100}',
        )
        data = json.loads(result)

        assert data["delivered"] is True
        body = mock_post.call_args.kwargs["json"]
        assert body["from_agent"] == "agent_a"
        assert body["payload"]["amount"] == 100

    def test_connect_agents_invalid_payload(self, agentid_tools: AgentIDTools) -> None:
        result = agentid_tools.connect_agents(
            from_agent="agent_a",
            to_agent="agent_b",
            payload="not valid json{{{",
        )
        data = json.loads(result)
        assert "error" in data
        assert "valid JSON" in data["error"]

    def test_connect_agents_no_api_key(self, agentid_tools_no_key: AgentIDTools) -> None:
        result = agentid_tools_no_key.connect_agents(
            from_agent="a",
            to_agent="b",
            payload="{}",
        )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Tool decorator verification
# ---------------------------------------------------------------------------

class TestToolDecorators:
    """Verify that the @tool decorator is applied correctly."""

    def test_verify_agent_is_marked_as_tool(self) -> None:
        assert getattr(AgentIDTools.verify_agent, "_upsonic_is_tool", False) is True

    def test_register_agent_is_marked_as_tool(self) -> None:
        assert getattr(AgentIDTools.register_agent, "_upsonic_is_tool", False) is True

    def test_discover_agents_is_marked_as_tool(self) -> None:
        assert getattr(AgentIDTools.discover_agents, "_upsonic_is_tool", False) is True

    def test_connect_agents_is_marked_as_tool(self) -> None:
        assert getattr(AgentIDTools.connect_agents, "_upsonic_is_tool", False) is True
