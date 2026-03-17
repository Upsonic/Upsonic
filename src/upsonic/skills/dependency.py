"""Skill dependency resolution — topological sort and cycle detection."""

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from .skill import Skill

logger = logging.getLogger(__name__)


def get_missing_dependencies(skills: Dict[str, "Skill"]) -> Dict[str, List[str]]:
    """Return a mapping of skill names to their missing dependency names.

    Only includes skills that have at least one missing dependency.
    """
    missing: Dict[str, List[str]] = {}
    for name, skill in skills.items():
        absent = [dep for dep in skill.dependencies if dep not in skills]
        if absent:
            missing[name] = absent
    return missing


def detect_cycles(skills: Dict[str, "Skill"]) -> List[List[str]]:
    """Detect dependency cycles using DFS with three-color marking.

    Returns:
        A list of cycles, where each cycle is a list of skill names
        forming the cycle path. Empty if no cycles exist.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {name: WHITE for name in skills}
    cycles: List[List[str]] = []
    path: List[str] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        path.append(node)

        for dep in skills[node].dependencies:
            if dep not in color:
                continue  # missing dependency — handled elsewhere
            if color[dep] == GRAY:
                # Found a cycle: extract from dep's position in path
                idx = path.index(dep)
                cycles.append(path[idx:] + [dep])
            elif color[dep] == WHITE:
                dfs(dep)

        path.pop()
        color[node] = BLACK

    for name in skills:
        if color[name] == WHITE:
            dfs(name)

    return cycles


def resolve_load_order(skills: Dict[str, "Skill"]) -> List[str]:
    """Topological sort of skills by their dependencies (Kahn's algorithm).

    Returns:
        An ordered list of skill names such that dependencies come before
        dependents.

    Raises:
        SkillValidationError: If a dependency cycle is detected.
    """
    from upsonic.utils.package.exception import SkillValidationError

    # Build adjacency and in-degree
    in_degree: Dict[str, int] = {name: 0 for name in skills}
    dependents: Dict[str, List[str]] = {name: [] for name in skills}

    for name, skill in skills.items():
        for dep in skill.dependencies:
            if dep in skills:
                in_degree[name] += 1
                dependents[dep].append(name)

    # Start with nodes that have no dependencies
    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(order) != len(skills):
        # Cycle detected — find and report it
        remaining = set(skills.keys()) - set(order)
        cycles = detect_cycles(
            {k: v for k, v in skills.items() if k in remaining}
        )
        cycle_strs = [" -> ".join(c) for c in cycles]
        raise SkillValidationError(
            "Dependency cycle detected",
            errors=[f"Cycle: {cs}" for cs in cycle_strs]
            if cycle_strs
            else [f"Cycle involving: {', '.join(remaining)}"],
        )

    return order
