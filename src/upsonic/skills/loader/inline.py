"""Inline/programmatic skill loader."""

from typing import List

from upsonic.utils.package.exception import SkillValidationError

from upsonic.skills.skill import Skill
from .base import SkillLoader


class InlineSkills(SkillLoader):
    """Load skills from pre-constructed :class:`Skill` objects.

    Useful for programmatically creating skills without filesystem structures::

        from upsonic.skills import Skill, Skills, InlineSkills

        skill = Skill(
            name="my-skill",
            description="A custom skill",
            instructions="Do this and that",
            source_path="",
        )
        skills = Skills(loaders=[InlineSkills([skill])])

    Args:
        skills: List of pre-built :class:`Skill` instances.
        validate: If ``True``, validate skill metadata (name/description rules).
    """

    def __init__(self, skills: List[Skill], validate: bool = False) -> None:
        self._skills = list(skills)
        self.validate = validate

    def load(self) -> List[Skill]:
        if self.validate:
            from upsonic.skills.validator import validate_metadata

            for skill in self._skills:
                meta = {"name": skill.name, "description": skill.description}
                if skill.dependencies:
                    meta["dependencies"] = skill.dependencies
                errors = validate_metadata(meta, skill_dir=None)
                if errors:
                    raise SkillValidationError(
                        f"Inline skill '{skill.name}' invalid", errors=errors
                    )
        return list(self._skills)
