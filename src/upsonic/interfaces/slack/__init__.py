from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .slack import SlackInterface
    from .schemas import SlackEventResponse, SlackChallengeResponse

def _get_slack_classes():
    """Lazy import of Slack classes."""
    from .slack import SlackInterface
    from .schemas import SlackEventResponse, SlackChallengeResponse
    
    return {
        'SlackInterface': SlackInterface,
        'SlackEventResponse': SlackEventResponse,
        'SlackChallengeResponse': SlackChallengeResponse,
    }

def __getattr__(name: str) -> Any:
    """Lazy loading of heavy modules and classes."""
    slack_classes = _get_slack_classes()
    if name in slack_classes:
        return slack_classes[name]
    
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Please import from the appropriate sub-module."
    )

__all__ = ["SlackInterface", "SlackEventResponse", "SlackChallengeResponse"]

