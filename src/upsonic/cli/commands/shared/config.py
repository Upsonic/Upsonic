import json
from pathlib import Path
from typing import Any, Dict, Optional

# Cache for config files to avoid repeated I/O
_CONFIG_CACHE = {}


def load_config(config_path: Path, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load and parse config file with caching.
    
    Args:
        config_path: Path to upsonic_configs.json
        use_cache: Whether to use cached config (default: True)
    
    Returns:
        Parsed config dictionary or None if error
    """
    cache_key = str(config_path.absolute())
    
    if use_cache and cache_key in _CONFIG_CACHE:
        # Check if file has been modified
        try:
            current_mtime = config_path.stat().st_mtime
            cached_mtime, cached_data = _CONFIG_CACHE[cache_key]
            if current_mtime == cached_mtime:
                return cached_data
        except Exception:
            pass
    
    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # Cache it
        if use_cache:
            try:
                mtime = config_path.stat().st_mtime
                _CONFIG_CACHE[cache_key] = (mtime, config_data)
            except Exception:
                pass
        
        return config_data
    except (FileNotFoundError, json.JSONDecodeError):
        return None

