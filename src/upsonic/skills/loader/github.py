"""GitHub remote skill loader — download skills from GitHub repositories."""

import io
import logging
import os
import tarfile
from pathlib import Path
from typing import List, Optional

from upsonic.utils.package.exception import SkillDownloadError

from .remote_base import RemoteSkillLoader

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


class GitHubSkills(RemoteSkillLoader):
    """Load skills from a GitHub repository.

    Downloads the repository tarball via the GitHub API and extracts
    the specified skill path.

    Args:
        repo: Repository in ``owner/name`` format (e.g. ``"upsonic/skills-library"``).
        branch: Git branch to fetch (default: ``"main"``).
        path: Path within the repository containing skills (default: ``"skills/"``).
        token: GitHub API token. Falls back to ``GITHUB_TOKEN`` / ``GH_TOKEN`` env vars.
        skills: Optional list of specific skill names to include.
        **kwargs: Passed to :class:`RemoteSkillLoader`.

    Example::

        loader = GitHubSkills(
            repo="upsonic/skills-library",
            path="skills/",
            branch="main",
        )
    """

    def __init__(
        self,
        repo: str,
        branch: str = "main",
        path: str = "skills/",
        token: Optional[str] = None,
        skills: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.repo = repo
        self.branch = branch
        self.path = path.rstrip("/")
        self.token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        self.skill_names = skills

    def _source_key(self) -> str:
        return f"github:{self.repo}:{self.branch}:{self.path}"

    def _download(self, target_dir: Path) -> None:
        try:
            import httpx
        except ImportError:
            raise SkillDownloadError(
                "httpx is required for GitHub skill loading. "
                "Install it with: pip install httpx"
            )

        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        url = f"https://api.github.com/repos/{self.repo}/tarball/{self.branch}"

        try:
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as resp:
                if resp.status_code != 200:
                    raise SkillDownloadError(
                        f"GitHub API returned status {resp.status_code} for {self.repo}"
                    )

                content = b""
                for chunk in resp.iter_bytes(chunk_size=8192):
                    content += chunk
                    if len(content) > _MAX_DOWNLOAD_SIZE:
                        raise SkillDownloadError(
                            f"Download exceeds {_MAX_DOWNLOAD_SIZE // (1024*1024)}MB limit"
                        )
        except httpx.HTTPError as e:
            raise SkillDownloadError(f"Failed to download from GitHub: {e}")

        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                # Find the root directory in the tarball (GitHub adds a prefix)
                members = tar.getmembers()
                if not members:
                    raise SkillDownloadError("Empty tarball from GitHub")

                root_prefix = members[0].name.split("/")[0]
                skill_prefix = f"{root_prefix}/{self.path}/"

                for member in members:
                    if not member.name.startswith(skill_prefix):
                        continue

                    # Security: reject path traversal and symlinks
                    relative = member.name[len(skill_prefix):]
                    if not relative or ".." in relative:
                        continue
                    if member.issym() or member.islnk():
                        logger.warning("Skipping symlink: %s", member.name)
                        continue

                    # Filter by skill names if specified
                    if self.skill_names:
                        top_dir = relative.split("/")[0]
                        if top_dir not in self.skill_names:
                            continue

                    dest = target_dir / relative
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as f:
                            if f is not None:
                                dest.write_bytes(f.read())
        except tarfile.TarError as e:
            raise SkillDownloadError(f"Failed to extract tarball: {e}")
