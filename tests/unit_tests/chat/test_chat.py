import asyncio
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import pytest

import upsonic.chat.chat as chat_module
from upsonic.chat.chat import Chat
from upsonic.chat.session_manager import SessionState
from upsonic.usage import RequestUsage


@dataclass
class MemoryStub:
    storage: object | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.storage = kwargs.get("storage")


@dataclass
class ChatMessageStub:
    content: str
    role: str
    timestamp: float
    attachments: List[str] | None = None

    @classmethod
    def from_model_message(cls, message):
        return cls(
            content=getattr(message, "text", ""),
            role="assistant",
            timestamp=time.time(),
        )


class DummyModelResponse:
    def __init__(self, text: str, usage: RequestUsage | None = None):
        self.text = text
        self.usage = usage
        self.kind = "response"
        self.parts = [SimpleNamespace(content=text)]


class FakeRunResult:
    def __init__(self, messages):
        self._messages = messages

    def new_messages(self):
        return self._messages


class FakeStreamResult:
    def __init__(self, chunks, messages):
        self._chunks = chunks
        self._messages = messages

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def stream_output(self):
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk

    def get_final_output(self):
        return "".join(self._chunks)

    def new_messages(self):
        return self._messages


class FakeAgent:
    def __init__(self):
        self.model = SimpleNamespace(name="fake-model")
        self.memory = None
        self.run_result = FakeRunResult([])
        self.stream_result = FakeStreamResult([], [])
        self.do_async_calls = []
        self.stream_async_calls = []

    async def do_async(self, task, **kwargs):
        self.do_async_calls.append((task, kwargs))
        return "agent-response"

    def get_run_result(self):
        return self.run_result

    async def stream_async(self, task, **kwargs):
        self.stream_async_calls.append((task, kwargs))
        return self.stream_result


@pytest.fixture(autouse=True)
def patch_chat_dependencies(monkeypatch):
    monkeypatch.setattr(chat_module, "Memory", MemoryStub)
    monkeypatch.setattr(chat_module, "ChatMessage", ChatMessageStub)
    monkeypatch.setattr(chat_module, "ModelResponse", DummyModelResponse)


def _make_response(text: str, usage: RequestUsage | None = None) -> DummyModelResponse:
    return DummyModelResponse(text=text, usage=usage)


@pytest.mark.asyncio
async def test_chat_initialization():
    agent = FakeAgent()
    chat = Chat(
        session_id=" session-123 ",
        user_id=" user-456 ",
        agent=agent,
        full_session_memory=False,
        summary_memory=True,
        user_analysis_memory=True,
        num_last_messages=5,
    )

    assert isinstance(agent.memory, MemoryStub)
    assert chat.session_id == "session-123"
    assert chat.user_id == "user-456"
    assert chat.state == SessionState.IDLE
    assert agent.memory.kwargs["summary_memory"] is True
    assert agent.memory.kwargs["user_analysis_memory"] is True


@pytest.mark.asyncio
async def test_chat_send_message():
    agent = FakeAgent()
    agent.run_result = FakeRunResult([_make_response("assistant")])
    chat = Chat(session_id="s1", user_id="u1", agent=agent)

    response = await chat.invoke("Hello there", attachments=["file.txt"])

    assert response == "agent-response"
    assert len(chat.all_messages) == 2  # user + assistant
    assert chat.all_messages[0].content == "Hello there"
    assert chat.all_messages[0].attachments == ["file.txt"]
    assert chat.all_messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_chat_get_response_updates_usage():
    agent = FakeAgent()
    usage = RequestUsage(input_tokens=10, output_tokens=4)
    agent.run_result = FakeRunResult([_make_response("Tokens", usage=usage)])
    chat = Chat(session_id="s2", user_id="u2", agent=agent)

    await chat.invoke("Count tokens")

    assert chat.input_tokens == 10
    assert chat.output_tokens == 4
    assert chat.total_cost >= 0  # cost tracker always non-negative


@pytest.mark.asyncio
async def test_chat_conversation_flow():
    agent = FakeAgent()
    agent.run_result = FakeRunResult([_make_response("First reply")])
    chat = Chat(session_id="flow", user_id="user", agent=agent)

    await chat.invoke("First message")
    agent.run_result = FakeRunResult([_make_response("Second reply")])
    await chat.invoke("Second message")

    assert len(chat.all_messages) == 4
    recent = chat.get_recent_messages(2)
    assert [msg.content for msg in recent] == ["Second message", "Second reply"]


@pytest.mark.asyncio
async def test_chat_streaming_updates_tokens():
    agent = FakeAgent()
    usage = RequestUsage(input_tokens=2, output_tokens=3)
    agent.stream_result = FakeStreamResult(
        ["chunk1", "chunk2"], [_make_response("Streamed", usage=usage)]
    )
    chat = Chat(session_id="stream", user_id="user", agent=agent)

    chunks = []
    stream_iter = await chat.invoke("Stream please", stream=True)
    async for chunk in stream_iter:
        chunks.append(chunk)

    assert "".join(chunks) == "chunk1chunk2"
    assert chat.input_tokens == 2
    assert chat.output_tokens == 3


@pytest.mark.asyncio
async def test_chat_session_management():
    agent = FakeAgent()
    agent.run_result = FakeRunResult([_make_response("Done")])
    chat = Chat(session_id="manage", user_id="user", agent=agent)

    await chat.invoke("Hello")
    chat.clear_history()
    assert chat.all_messages == []

    chat.reset_session()
    assert chat.state == SessionState.IDLE

    class StorageStub:
        def __init__(self):
            self.disconnected = False

        async def is_connected_async(self):
            return not self.disconnected

        async def disconnect_async(self):
            self.disconnected = True

    storage_stub = StorageStub()
    chat._storage = storage_stub

    await chat.close()
    assert storage_stub.disconnected is True
