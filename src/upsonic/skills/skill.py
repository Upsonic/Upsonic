"""Skill data model — a pure data container for loaded skill information."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Skill:
    """Represents a skill that an agent can use.

    A skill provides structured instructions, reference documentation,
    and optional scripts that an agent can access to perform specific tasks.

    This is a data-only class — loading, parsing and validation are handled
    by :class:`~upsonic.skills.loader.LocalSkills` and
    :mod:`~upsonic.skills.validator`.

    Attributes:
        name: Unique skill name (from folder name or SKILL.md frontmatter).
        description: Short description of what the skill does.
        instructions: Full SKILL.md body (the guidance for the agent).
        source_path: Filesystem path to the skill folder.
        scripts: List of script filenames in ``scripts/``.
        references: List of reference filenames in ``references/``.
        assets: List of asset filenames in ``assets/`` (templates, fonts, icons).
        metadata: Optional metadata from frontmatter (version, author, tags …).
        license: Optional license information.
        compatibility: Optional compatibility requirements.
        allowed_tools: Optional list of tools this skill is allowed to use.
    """

    name: str
    description: str
    instructions: str
    source_path: str
    scripts: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    assets: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: Optional[List[str]] = None
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


    def to_dict(self) -> Dict[str, Any]:
        """Convert the Skill to a dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "source_path": self.source_path,
            "scripts": self.scripts,
            "references": self.references,
            "assets": self.assets,
            "metadata": self.metadata,
            "license": self.license,
            "compatibility": self.compatibility,
            "allowed_tools": self.allowed_tools,
            "version": self.version,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create a Skill from a dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            instructions=data["instructions"],
            source_path=data["source_path"],
            scripts=data.get("scripts", []),
            references=data.get("references", []),
            assets=data.get("assets", []),
            metadata=data.get("metadata"),
            license=data.get("license"),
            compatibility=data.get("compatibility"),
            allowed_tools=data.get("allowed_tools"),
            version=data.get("version"),
            dependencies=data.get("dependencies", []),
        )

    def __repr__(self) -> str:
        return (
            f"Skill(name='{self.name}', "
            f"scripts={len(self.scripts)}, "
            f"references={len(self.references)}, "
            f"assets={len(self.assets)})"
        )
