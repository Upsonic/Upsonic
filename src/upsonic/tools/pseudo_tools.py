from __future__ import annotations
from typing import TYPE_CHECKING

from upsonic.tools.thought import Thought


def plan_and_execute(thought: Thought) -> str:
    """
    The master tool for complex tasks. Call this tool first with your
    structured reasoning and a multi-step plan. This tool will then take
    control, execute each tool in your plan sequentially, and provide you
    with the full history of results to synthesize a final answer.
    """
    pass