"""Base class for remote skill loaders with local caching."""

import abc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from upsonic.skills.skill import Skill
from .base import SkillLoader
from .local import LocalSkills

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".upsonic" / "skills_cache"


class RemoteSkillLoader(SkillLoader):
    """Abstract base for remote skill loaders with local disk caching.

    Subclasses implement :meth:`_download` to fetch skills to a local directory.
    The base class handles TTL-based caching and delegation to :class:`LocalSkills`.

    Args:
        cache_dir: Directory for cached downloads. Defaults to ``~/.upsonic/skills_cache/``.
        cache_ttl: Cache time-to-live in seconds (default: 3600 = 1 hour).
        validate: Whether to validate loaded skills.
        force_refresh: If ``True``, bypass cache and always re-download.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600,
        validate: bool = True,
        force_refresh: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_ttl = cache_ttl
        self.validate = validate
        self.force_refresh = force_refresh

    @abc.abstractmethod
    def _download(self, target_dir: Path) -> None:
        """Download skill files into *target_dir*.

        Raises:
            SkillDownloadError: If the download fails.
        """
        ...

    @abc.abstractmethod
    def _source_key(self) -> str:
        """Return a unique string identifying this remote source (used for cache path)."""
        ...

    def load(self) -> List[Skill]:
        cache_path = self._get_cache_path()
        meta_file = cache_path / ".cache_meta.json"

        if not self.force_refresh and self._is_cache_fresh(meta_file):
            logger.debug("Using cached skills from %s", cache_path)
        else:
            logger.debug("Downloading skills to %s", cache_path)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._download(cache_path)
            self._write_cache_meta(meta_file)

        return LocalSkills(str(cache_path), validate=self.validate).load()

    def _get_cache_path(self) -> Path:
        source_hash = hashlib.sha256(self._source_key().encode()).hexdigest()[:16]
        loader_type = type(self).__name__.lower()
        return self.cache_dir / loader_type / source_hash

    def _is_cache_fresh(self, meta_file: Path) -> bool:
        if not meta_file.exists():
            return False
        try:
            meta = json.loads(meta_file.read_text())
            cached_at = meta.get("cached_at", 0)
            return (time.time() - cached_at) < self.cache_ttl
        except Exception:
            return False

    def _write_cache_meta(self, meta_file: Path) -> None:
        meta = {
            "cached_at": time.time(),
            "source": self._source_key(),
            "loader": type(self).__name__,
        }
        meta_file.write_text(json.dumps(meta, indent=2))
