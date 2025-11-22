import time
from upsonic.chat.session_manager import SessionManager, SessionState, SessionMetrics
from upsonic.chat.message import ChatMessage


def test_session_manager_initialization():
    """Test SessionManager initialization."""
    manager = SessionManager(
        session_id="test-session-1",
        user_id="test-user-1",
        debug=False,
        max_concurrent_invocations=2,
    )

    assert manager.session_id == "test-session-1"
    assert manager.user_id == "test-user-1"
    assert manager.debug is False
    assert manager.state == SessionState.IDLE
    assert manager._max_concurrent_invocations == 2
    assert manager._concurrent_invocations == 0
    assert len(manager.all_messages) == 0


def test_session_manager_create_session():
    """Test session creation."""
    manager = SessionManager(session_id="test-session-2", user_id="test-user-2")

    assert manager.session_id == "test-session-2"
    assert manager.user_id == "test-user-2"
    assert manager.is_session_active() is True
    assert manager.get_message_count() == 0


def test_session_manager_get_session():
    """Test session retrieval."""
    manager = SessionManager(session_id="test-session-3", user_id="test-user-3")

    # Test getting session metrics
    metrics = manager.get_session_metrics()
    assert isinstance(metrics, SessionMetrics)
    assert metrics.session_id == "test-session-3"
    assert metrics.user_id == "test-user-3"

    # Test getting session stats
    stats = manager.get_session_stats()
    assert stats["session_id"] == "test-session-3"
    assert stats["user_id"] == "test-user-3"
    assert stats["state"] == SessionState.IDLE.value


def test_session_manager_delete_session():
    """Test session deletion."""
    manager = SessionManager(session_id="test-session-4", user_id="test-user-4")

    # Add some messages
    message = ChatMessage(content="Test message", role="user", timestamp=time.time())
    manager.add_message(message)

    # Close session
    manager.close_session()

    assert manager.state == SessionState.IDLE
    assert manager._concurrent_invocations == 0
    assert manager._metrics.end_time is not None


def test_session_manager_list_sessions():
    """Test listing sessions."""
    manager = SessionManager(session_id="test-session-5", user_id="test-user-5")

    # Test session summary
    summary = manager.get_session_summary()
    assert "Session Summary" in summary
    assert "Duration" in summary
    assert "Messages" in summary

    # Test debug info
    debug_info = manager.get_debug_info()
    assert debug_info["session_id"] == "test-session-5"
    assert debug_info["user_id"] == "test-user-5"
    assert "state" in debug_info
    assert "concurrent_invocations" in debug_info
