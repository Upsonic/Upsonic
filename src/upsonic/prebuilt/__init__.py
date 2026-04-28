"""
Upsonic prebuilt agents.

Ready-to-use autonomous agents whose system prompt, first-message template,
and skills are bundled inside this package. Each prebuilt lives in its own
subfolder (e.g. ``applied_scientist/``) with the source code in
``agent.py`` and the agent's prompt template in ``template/``.

Every prebuilt class is a subclass of
:class:`~upsonic.prebuilt.prebuilt_agent_base.PrebuiltAutonomousAgentBase`,
which clones the template folder into the user's workspace on every run.

Usage:
    ```python
    from upsonic.prebuilt import AppliedScientist

    scientist = AppliedScientist(model="openai/gpt-4o", workspace="./ws")
    exp = scientist.new_experiment(
        name="tabpfn_adult",
        # Anything: local path, URL, git / Kaggle / arXiv / HF link,
        # or a free-text idea like "swap XGBoost for CatBoost".
        research_source="example_1/tabpfn.pdf",
        current_notebook="example_1/baseline.ipynb",
        # `current_data`, `experiments_directory`, and `inputs` are all
        # optional. When `current_data` is omitted, the agent reads the
        # notebook itself and infers the data source from its loading
        # cells. `experiments_directory` defaults to "./experiments".
    )
    exp.run()
    ```

See ``Docs/ai/new_prebuilt_agent_adding.md`` for instructions on how to
contribute a new prebuilt agent.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .prebuilt_agent_base import PrebuiltAutonomousAgentBase
    from .applied_scientist.agent import (
        AppliedScientist,
        Experiment,
        ExperimentResult,
    )


def _get_classes() -> dict[str, Any]:
    from .prebuilt_agent_base import PrebuiltAutonomousAgentBase
    from .applied_scientist.agent import (
        AppliedScientist,
        Experiment,
        ExperimentResult,
    )
    return {
        "PrebuiltAutonomousAgentBase": PrebuiltAutonomousAgentBase,
        "AppliedScientist": AppliedScientist,
        "Experiment": Experiment,
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
    "PrebuiltAutonomousAgentBase",
    "AppliedScientist",
    "Experiment",
    "ExperimentResult",
]
