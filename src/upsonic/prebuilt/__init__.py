"""
Upsonic prebuilt agents.

Ready-to-use autonomous agents backed by templates in the
``Upsonic/AutonomousAgents`` repository. Each class auto-wires the repo and
folder and exposes a template-aware API on top of
:class:`~upsonic.agent.prebuilt_autonomous_agent.PrebuiltAutonomousAgent`.

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
        current_data="downloaded in notebook (ucimlrepo, id=2)",
        # `experiments_directory` and `inputs` are both optional —
        # experiments_directory defaults to "./experiments" and inputs is
        # auto-derived from the arguments above.
    )
    exp.run()
    ```
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .upsonic_prebuilt_agents import (
        AppliedScientist,
        Experiment,
        ExperimentResult,
    )


def _get_classes() -> dict[str, Any]:
    from .upsonic_prebuilt_agents import (
        AppliedScientist,
        Experiment,
        ExperimentResult,
    )
    return {
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
    "AppliedScientist",
    "Experiment",
    "ExperimentResult",
]
