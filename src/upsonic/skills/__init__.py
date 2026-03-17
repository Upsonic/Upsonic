"""Skills module — structured domain expertise for agents.

Skills are self-contained packages that extend agent capabilities through
instructions, scripts, and reference documentation.  They enable progressive
discovery: agents see skill summaries upfront and load details on demand.

Example::

    from upsonic import Agent
    from upsonic.skills import Skills, LocalSkills

    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        skills=Skills(loaders=[LocalSkills("/path/to/skills")])
    )
"""

from upsonic.utils.package.exception import (
    SkillError,
    SkillParseError,
    SkillValidationError,
)

from .loader import (
    BuiltinSkills,
    GitHubSkills,
    InlineSkills,
    LocalSkills,
    RemoteSkillLoader,
    SkillLoader,
    URLSkills,
)
from .metrics import SkillMetrics
from .skill import Skill
from .skills import Skills

__all__ = [
    "Skill",
    "Skills",
    "SkillLoader",
    "LocalSkills",
    "InlineSkills",
    "BuiltinSkills",
    "RemoteSkillLoader",
    "GitHubSkills",
    "URLSkills",
    "SkillMetrics",
    "SkillError",
    "SkillParseError",
    "SkillValidationError",
]
