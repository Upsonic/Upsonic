"""Local filesystem skill loader."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from upsonic.utils.package.exception import SkillValidationError

from upsonic.skills.skill import Skill
from .base import SkillLoader

logger = logging.getLogger(__name__)


class LocalSkills(SkillLoader):
    """Load skills from local filesystem directories.

    Supports two patterns:

    1. Path to a **single** skill directory (contains ``SKILL.md`` directly).
    2. Path to a **parent** directory containing multiple skill subdirectories.

    Args:
        path: Filesystem path to a skill folder or a directory of skill folders.
        validate: If ``True`` (default), skills are validated against the spec.
            Set to ``False`` during development to skip strict checks.

    Examples::

        # Load a single skill
        loader = LocalSkills("/path/to/skills/code-review")

        # Load all skills in a directory
        loader = LocalSkills("/path/to/skills")

        # Skip validation during development
        loader = LocalSkills("/path/to/skills", validate=False)
    """

    def __init__(
        self,
        path: str,
        validate: bool = True,
        version_constraint: Optional[str] = None,
    ) -> None:
        self.path = Path(path).resolve()
        self.validate = validate
        self.version_constraint = version_constraint
        if not self.path.exists():
            raise FileNotFoundError(f"Skills path does not exist: {self.path}")

    def load(self) -> List[Skill]:
        skills: List[Skill] = []

        if (self.path / "SKILL.md").is_file():
            # Single skill folder
            skill = self._load_skill_from_folder(self.path)
            if skill:
                skills.append(skill)
        else:
            # Parent directory containing skill folders
            for item in sorted(self.path.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    if (item / "SKILL.md").is_file():
                        skill = self._load_skill_from_folder(item)
                        if skill:
                            skills.append(skill)
                    else:
                        logger.debug("Skipping directory without SKILL.md: %s", item)

        if self.version_constraint:
            from upsonic.skills.version import SkillVersion, VersionConstraint

            vc = VersionConstraint(self.version_constraint)
            filtered: List[Skill] = []
            for skill in skills:
                if skill.version:
                    try:
                        sv = SkillVersion.parse(skill.version)
                        if vc.satisfies(sv):
                            filtered.append(skill)
                        else:
                            logger.debug(
                                "Skill '%s' version %s does not satisfy %s, skipping",
                                skill.name,
                                skill.version,
                                self.version_constraint,
                            )
                    except ValueError:
                        logger.warning(
                            "Skill '%s' has invalid version '%s', including anyway",
                            skill.name,
                            skill.version,
                        )
                        filtered.append(skill)
                else:
                    logger.debug(
                        "Skill '%s' has no version, including despite constraint",
                        skill.name,
                    )
                    filtered.append(skill)
            skills = filtered

        logger.debug("Loaded %d skills from %s", len(skills), self.path)
        return skills


    def _load_skill_from_folder(self, folder: Path) -> Optional[Skill]:
        """Load a single skill from *folder*.

        Raises:
            SkillValidationError: When validation is enabled and the skill is
                invalid.
        """
        if self.validate:
            from upsonic.skills.validator import validate_skill_directory

            errors = validate_skill_directory(folder)
            if errors:
                raise SkillValidationError(
                    f"Skill validation failed for '{folder.name}'",
                    errors=errors,
                )

        skill_md = folder / "SKILL.md"
        try:
            content = skill_md.read_text(encoding="utf-8")
            frontmatter, instructions = self._parse_skill_md(content)

            # Support version as top-level field or nested under metadata
            version = frontmatter.get("version")
            if version is None:
                metadata_dict = frontmatter.get("metadata") or {}
                version = metadata_dict.get("version") if isinstance(metadata_dict, dict) else None
            deps = frontmatter.get("dependencies")
            if not isinstance(deps, list):
                deps = []

            return Skill(
                name=frontmatter.get("name", folder.name),
                description=frontmatter.get("description", ""),
                instructions=instructions,
                source_path=str(folder),
                scripts=self._discover_files(folder / "scripts"),
                references=self._discover_files(folder / "references"),
                assets=self._discover_files(folder / "assets"),
                metadata=frontmatter.get("metadata"),
                license=frontmatter.get("license"),
                compatibility=frontmatter.get("compatibility"),
                allowed_tools=frontmatter.get("allowed-tools"),
                version=version,
                dependencies=deps,
            )
        except SkillValidationError:
            raise
        except Exception as e:
            logger.warning("Error loading skill from %s: %s", folder, e)
            return None

    def _parse_skill_md(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse SKILL.md content into ``(frontmatter_dict, instructions_body)``."""
        frontmatter: Dict[str, Any] = {}
        instructions = content

        match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL
        )
        if not match:
            return frontmatter, instructions

        frontmatter_text = match.group(1)
        instructions = match.group(2).strip()

        try:
            import yaml

            frontmatter = yaml.safe_load(frontmatter_text) or {}
        except ImportError:
            frontmatter = self._parse_simple_frontmatter(frontmatter_text)
        except Exception as e:
            logger.warning("Error parsing YAML frontmatter: %s", e)
            frontmatter = self._parse_simple_frontmatter(frontmatter_text)

        return frontmatter, instructions

    @staticmethod
    def _parse_simple_frontmatter(text: str) -> Dict[str, Any]:
        """Minimal fallback parser for ``key: value`` pairs."""
        result: Dict[str, Any] = {}
        for line in text.strip().splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if value.startswith("[") and value.endswith("]"):
                    result[key] = [
                        v.strip().strip('"').strip("'")
                        for v in value[1:-1].split(",")
                        if v.strip()
                    ]
                elif value:
                    result[key] = value
        return result

    @staticmethod
    def _discover_files(directory: Path) -> List[str]:
        """Return sorted filenames from *directory*, ignoring hidden files."""
        if not directory.is_dir():
            return []
        return sorted(
            f.name
            for f in directory.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
