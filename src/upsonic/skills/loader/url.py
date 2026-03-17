"""URL remote skill loader — download skills from a URL archive."""

import io
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

from upsonic.utils.package.exception import SkillDownloadError

from .remote_base import RemoteSkillLoader

logger = logging.getLogger(__name__)

_MAX_DOWNLOAD_SIZE = 100 * 1024 * 1024  # 100 MB


class URLSkills(RemoteSkillLoader):
    """Load skills from a URL pointing to a .tar.gz or .zip archive.

    Args:
        url: URL to the skill archive.
        headers: Optional HTTP headers for the request.
        max_size: Maximum download size in bytes (default: 100MB).
        **kwargs: Passed to :class:`RemoteSkillLoader`.

    Example::

        loader = URLSkills("https://example.com/skills.tar.gz")
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        max_size: int = _MAX_DOWNLOAD_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.headers = headers or {}
        self.max_size = max_size

    def _source_key(self) -> str:
        return f"url:{self.url}"

    def _download(self, target_dir: Path) -> None:
        try:
            import httpx
        except ImportError:
            raise SkillDownloadError(
                "httpx is required for URL skill loading. "
                "Install it with: pip install httpx"
            )

        try:
            with httpx.stream(
                "GET", self.url, headers=self.headers,
                follow_redirects=True, timeout=60,
            ) as resp:
                if resp.status_code != 200:
                    raise SkillDownloadError(
                        f"HTTP {resp.status_code} from {self.url}"
                    )

                content = b""
                for chunk in resp.iter_bytes(chunk_size=8192):
                    content += chunk
                    if len(content) > self.max_size:
                        raise SkillDownloadError(
                            f"Download exceeds {self.max_size // (1024*1024)}MB limit"
                        )
        except httpx.HTTPError as e:
            raise SkillDownloadError(f"Failed to download from URL: {e}")

        # Detect archive type and extract
        if self.url.endswith(".zip") or content[:4] == b"PK\x03\x04":
            self._extract_zip(content, target_dir)
        else:
            self._extract_tar(content, target_dir)

    def _extract_tar(self, content: bytes, target_dir: Path) -> None:
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
                for member in tar.getmembers():
                    # Security checks
                    if ".." in member.name or member.name.startswith("/"):
                        logger.warning("Skipping unsafe path: %s", member.name)
                        continue
                    if member.issym() or member.islnk():
                        logger.warning("Skipping symlink: %s", member.name)
                        continue

                    dest = target_dir / member.name
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as f:
                            if f is not None:
                                dest.write_bytes(f.read())
        except tarfile.TarError as e:
            raise SkillDownloadError(f"Failed to extract tar archive: {e}")

    def _extract_zip(self, content: bytes, target_dir: Path) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for info in zf.infolist():
                    # Security checks
                    if ".." in info.filename or info.filename.startswith("/"):
                        logger.warning("Skipping unsafe path: %s", info.filename)
                        continue

                    dest = target_dir / info.filename
                    if info.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(info))
        except zipfile.BadZipFile as e:
            raise SkillDownloadError(f"Failed to extract zip archive: {e}")
