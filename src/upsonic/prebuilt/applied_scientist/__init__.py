"""
AppliedScientist prebuilt autonomous agent.

The agent's system prompt, first-message template, and skills live under
``template/`` and are pulled into the runtime workspace at run time by
:class:`~upsonic.prebuilt.prebuilt_agent_base.PrebuiltAutonomousAgentBase`.

Usage:
    ```python
    from upsonic.prebuilt import AppliedScientist

    scientist = AppliedScientist(model="openai/gpt-4o", workspace="./ws")
    exp = scientist.new_experiment(
        name="tabpfn_adult",
        research_source="example_1/tabpfn.pdf",
        current_notebook="example_1/baseline.ipynb",
    )
    exp.run()
    ```
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import (
        AppliedScientist,
        Experiment,
        ExperimentRecord,
        ExperimentRegistry,
        ExperimentResult,
    )


def _get_classes() -> dict[str, Any]:
    from .agent import (
        AppliedScientist,
        Experiment,
        ExperimentRecord,
        ExperimentRegistry,
        ExperimentResult,
    )
    return {
        "AppliedScientist": AppliedScientist,
        "Experiment": Experiment,
        "ExperimentRecord": ExperimentRecord,
        "ExperimentRegistry": ExperimentRegistry,
        "ExperimentResult": ExperimentResult,
    }


def __getattr__(name: str) -> Any:
    classes = _get_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(
        f"module '{__name__}' has no attribute '{name}'. "
        f"Available: {list(classes.keys())}"
    )


__all__ = [
    "AppliedScientist",
    "Experiment",
    "ExperimentRecord",
    "ExperimentRegistry",
    "ExperimentResult",
]
