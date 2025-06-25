# context/builder.py
from typing import Iterable, List
from .default_prompt import default_prompt
from .registry import REGISTRY
from .strategy import ContextStrategy


def _select_strategy(obj, strategies: List[ContextStrategy]) -> ContextStrategy:
    for strategy in strategies:
        if strategy.matches(obj):
            return strategy
    raise ValueError(f"No strategy registered for type {type(obj)}")


def build_context(objects: Iterable, extra_strategies: List[ContextStrategy] | None = None) -> str:
    strategies = REGISTRY + (extra_strategies or [])

    stream: List[str] = ["<Context>"]

    # guarantee the default prompt is always present
    objects = list(objects) + [default_prompt()]

    for obj in objects:
        stream.append(_select_strategy(obj, strategies).format(obj))

    stream.append("</Context>")
    return "\n".join(stream)
