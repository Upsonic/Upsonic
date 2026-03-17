"""Skill versioning — parse and compare semantic versions."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SkillVersion:
    """Represents a semantic version (major.minor.patch)."""

    major: int
    minor: int
    patch: int = 0

    @classmethod
    def parse(cls, s: str) -> "SkillVersion":
        """Parse a version string like ``1.2.3`` or ``1.0``.

        Raises:
            ValueError: If the string is not a valid version.
        """
        s = s.strip()
        parts = s.split(".")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(
                f"Invalid version string '{s}': expected MAJOR.MINOR or MAJOR.MINOR.PATCH"
            )
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2]) if len(parts) == 3 else 0
        except ValueError:
            raise ValueError(f"Invalid version string '{s}': parts must be integers")

        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"Invalid version string '{s}': parts must be non-negative")

        return cls(major=major, minor=minor, patch=patch)

    def _as_tuple(self) -> tuple:
        return (self.major, self.minor, self.patch)

    def __lt__(self, other: "SkillVersion") -> bool:
        return self._as_tuple() < other._as_tuple()

    def __le__(self, other: "SkillVersion") -> bool:
        return self._as_tuple() <= other._as_tuple()

    def __gt__(self, other: "SkillVersion") -> bool:
        return self._as_tuple() > other._as_tuple()

    def __ge__(self, other: "SkillVersion") -> bool:
        return self._as_tuple() >= other._as_tuple()

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


# Regex for a single constraint segment like ">=1.0.0" or "<2.0"
_CONSTRAINT_RE = re.compile(r"^(>=|<=|>|<|==|!=)(.+)$")


class _SingleConstraint:
    """A single version constraint like ``>=1.0.0``."""

    def __init__(self, op: str, version: SkillVersion) -> None:
        self.op = op
        self.version = version

    def satisfies(self, v: SkillVersion) -> bool:
        if self.op == ">=":
            return v >= self.version
        elif self.op == "<=":
            return v <= self.version
        elif self.op == ">":
            return v > self.version
        elif self.op == "<":
            return v < self.version
        elif self.op == "==":
            return v == self.version
        elif self.op == "!=":
            return v != self.version
        return False


class VersionConstraint:
    """Compound version constraint supporting comma-separated segments.

    Examples::

        VersionConstraint(">=1.0.0,<2.0.0")
        VersionConstraint("==1.2.3")
        VersionConstraint(">=1.0")

    All segments must be satisfied for :meth:`satisfies` to return ``True``.
    """

    def __init__(self, constraint_str: str) -> None:
        self._raw = constraint_str
        self._constraints: List[_SingleConstraint] = []
        for segment in constraint_str.split(","):
            segment = segment.strip()
            if not segment:
                continue
            m = _CONSTRAINT_RE.match(segment)
            if not m:
                raise ValueError(
                    f"Invalid version constraint segment: '{segment}'. "
                    "Expected format: OP VERSION (e.g. '>=1.0.0')"
                )
            op = m.group(1)
            ver = SkillVersion.parse(m.group(2))
            self._constraints.append(_SingleConstraint(op, ver))

        if not self._constraints:
            raise ValueError(f"Empty version constraint: '{constraint_str}'")

    def satisfies(self, version: SkillVersion) -> bool:
        """Return ``True`` if *version* satisfies all constraint segments."""
        return all(c.satisfies(version) for c in self._constraints)

    def __repr__(self) -> str:
        return f"VersionConstraint('{self._raw}')"
