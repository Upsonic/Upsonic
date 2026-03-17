"""Abstract base class for all skill loaders."""

import abc
from typing import List

from upsonic.skills.skill import Skill


class SkillLoader(abc.ABC):
    """Abstract base class for skill loaders."""

    @abc.abstractmethod
    def load(self) -> List[Skill]:
        """Load and return a list of :class:`Skill` objects.

        Raises:
            SkillValidationError: If any skill fails validation.
        """
        ...
