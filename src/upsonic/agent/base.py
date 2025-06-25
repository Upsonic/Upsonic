from __future__ import annotations

from abc import ABC


class AgentBase(ABC):
    """Lightweight mix-in that identifies an object as an Upsonic agent.

    This class purposefully contains **no logic or heavy imports** to avoid
    circular-dependency problems. Concrete agent implementations (e.g.
    ``Direct``) should inherit from this to be picked up by context
    strategies.
    """

    # Optional typed attributes that higher-level utilities may access.
    agent_id: str | None
    name: str | None
    company_url: str | None
    company_objective: str | None
    company_description: str | None
    system_prompt: str | None
