import json
import os
from pathlib import Path
from typing import Any, Optional


class Configuration:
    """
    Simple file-based configuration storage for package-level settings.
    Stores data in a JSON file in the user's home directory.
    """

    _config_dir = Path.home() / ".upsonic"
    _config_file = _config_dir / "config.json"
    _cache: Optional[dict] = None

    @classmethod
    def _ensure_config_dir(cls) -> None:
        """Ensure the configuration directory exists."""
        cls._config_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _load_config(cls) -> dict:
        """Load configuration from file."""
        if cls._cache is not None:
            return cls._cache

        cls._ensure_config_dir()

        if cls._config_file.exists():
            try:
                with open(cls._config_file, 'r') as f:
                    cls._cache = json.load(f)
                    return cls._cache
            except (json.JSONDecodeError, IOError):
                cls._cache = {}
                return cls._cache

        cls._cache = {}
        return cls._cache

    @classmethod
    def _save_config(cls, config: dict) -> None:
        """Save configuration to file."""
        cls._ensure_config_dir()

        try:
            with open(cls._config_file, 'w') as f:
                json.dump(config, f, indent=2)
            cls._cache = config
        except IOError as e:
            raise IOError(f"Failed to save configuration: {e}")

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The configuration value or default if not found
        """
        config = cls._load_config()
        return config.get(key, default)

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key to set
            value: The value to store
        """
        config = cls._load_config()
        config[key] = value
        cls._save_config(config)

    @classmethod
    def delete(cls, key: str) -> None:
        """
        Delete a configuration value.

        Args:
            key: The configuration key to delete
        """
        config = cls._load_config()
        if key in config:
            del config[key]
            cls._save_config(config)

    @classmethod
    def clear(cls) -> None:
        """Clear all configuration values."""
        cls._save_config({})

    @classmethod
    def all(cls) -> dict:
        """
        Get all configuration values.

        Returns:
            Dictionary of all configuration key-value pairs
        """
        return cls._load_config().copy()
