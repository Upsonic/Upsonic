"""Loader for built-in skills that ship with Upsonic."""

import importlib.resources
import logging
from pathlib import Path
from typing import List, Optional

from upsonic.skills.skill import Skill
from .base import SkillLoader
from .local import LocalSkills

logger = logging.getLogger(__name__)


class BuiltinSkills(SkillLoader):
    """Load skills from Upsonic's built-in skill library.

    Built-in skills are packaged with Upsonic and available without any
    filesystem paths or downloads.

    Args:
        skills: Optional list of skill names to load. If ``None``, all
            built-in skills are loaded.
        validate: Whether to validate skills (default: ``False`` since
            built-ins are pre-validated).

    Example::

        from upsonic.skills import Skills
        from upsonic.skills.builtins import BuiltinSkills

        skills = Skills(loaders=[BuiltinSkills()])
        # or load specific ones:
        skills = Skills(loaders=[BuiltinSkills(skills=["code-review"])])
    """

    def __init__(
        self,
        skills: Optional[List[str]] = None,
        validate: bool = False,
    ) -> None:
        self.skill_filter = skills
        self.validate = validate

    def load(self) -> List[Skill]:
        builtins_path = self._get_builtins_path()
        if builtins_path is None:
            logger.warning("Could not locate built-in skills directory")
            return []

        loader = LocalSkills(str(builtins_path), validate=self.validate)
        all_skills = loader.load()

        if self.skill_filter:
            return [s for s in all_skills if s.name in self.skill_filter]
        return all_skills

    def available_skills(self) -> List[str]:
        """Return names of all available built-in skills."""
        builtins_path = self._get_builtins_path()
        if builtins_path is None:
            return []
        return sorted(
            d.name
            for d in builtins_path.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and not d.name.startswith("_")
            and (d / "SKILL.md").exists()
        )

    @staticmethod
    def _get_builtins_path() -> Optional[Path]:
        """Locate the builtins directory, supporting both installed and dev layouts."""
        # Try importlib.resources first (works for installed packages)
        try:
            ref = importlib.resources.files("upsonic.skills.builtins")
            path = Path(str(ref))
            if path.is_dir():
                return path
        except (TypeError, ModuleNotFoundError):
            pass

        # Fallback: relative to this file (go up to skills/, then into builtins/)
        this_dir = Path(__file__).parent.parent / "builtins"
        if this_dir.is_dir():
            return this_dir

        return None
