"""Skill usage metrics — track how skills are accessed by agents."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SkillMetrics:
    """Usage metrics for a single skill.

    Attributes:
        load_count: Number of times instructions were loaded.
        reference_access_count: Number of reference document accesses.
        script_execution_count: Number of script executions.
        total_chars_loaded: Total characters loaded across all access types.
        last_used_timestamp: Unix timestamp of the last access.
    """

    load_count: int = 0
    reference_access_count: int = 0
    script_execution_count: int = 0
    total_chars_loaded: int = 0
    last_used_timestamp: Optional[float] = None

    def record_load(self, chars: int = 0) -> None:
        self.load_count += 1
        self.total_chars_loaded += chars
        self.last_used_timestamp = time.time()

    def record_reference_access(self, chars: int = 0) -> None:
        self.reference_access_count += 1
        self.total_chars_loaded += chars
        self.last_used_timestamp = time.time()

    def record_script_execution(self) -> None:
        self.script_execution_count += 1
        self.last_used_timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "load_count": self.load_count,
            "reference_access_count": self.reference_access_count,
            "script_execution_count": self.script_execution_count,
            "total_chars_loaded": self.total_chars_loaded,
            "last_used_timestamp": self.last_used_timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetrics":
        return cls(
            load_count=data.get("load_count", 0),
            reference_access_count=data.get("reference_access_count", 0),
            script_execution_count=data.get("script_execution_count", 0),
            total_chars_loaded=data.get("total_chars_loaded", 0),
            last_used_timestamp=data.get("last_used_timestamp"),
        )
