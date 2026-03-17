"""Skill loaders — discover and load skills from various sources."""

from .base import SkillLoader
from .builtin import BuiltinSkills
from .github import GitHubSkills
from .inline import InlineSkills
from .local import LocalSkills
from .remote_base import RemoteSkillLoader
from .url import URLSkills

__all__ = [
    "SkillLoader",
    "LocalSkills",
    "InlineSkills",
    "BuiltinSkills",
    "RemoteSkillLoader",
    "GitHubSkills",
    "URLSkills",
]
