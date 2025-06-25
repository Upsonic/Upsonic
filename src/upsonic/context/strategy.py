# context/strategy.py
from __future__ import annotations
from abc import ABC, abstractmethod


class ContextStrategy(ABC):
    """One‐function policy: does this strategy know how to handle 'obj'?"""

    @abstractmethod
    def matches(self, obj) -> bool:
        ...

    @abstractmethod
    def format(self, obj) -> str:
        ...
