import json
import hashlib
from pathlib import Path

from upsonic.messages.messages import ModelMessagesTypeAdapter


# Memory directory - same as current file location
MEMORY_DIR = Path(__file__).parent


def _get_agent_file_path(agent_id: str) -> Path:
    """Generate a unique file path for an agent using SHA256 hash."""
    agent_hash = hashlib.sha256(agent_id.encode('utf-8')).hexdigest()
    return MEMORY_DIR / f"{agent_hash}.json"


def save_agent_memory(agent, answer):
    """Save agent memory from pydantic response."""
    history = answer.all_messages()
    # Use ModelMessagesTypeAdapter to properly serialize bytes as base64
    json_data = ModelMessagesTypeAdapter.dump_python(history, mode='json')
    
    agent_file = _get_agent_file_path(agent.get_agent_id())
    
    try:
        with open(agent_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    except (OSError, UnicodeEncodeError):
        pass  # Silently fail if we can't write to file


def get_agent_memory(agent):
    """Get agent memory as pydantic messages."""
    agent_file = _get_agent_file_path(agent.get_agent_id())
    
    try:
        if not agent_file.exists():
            return []
        
        with open(agent_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if isinstance(json_data, list):
                return ModelMessagesTypeAdapter.validate_python(json_data)
            else:
                return []
    except (json.JSONDecodeError, FileNotFoundError, OSError, UnicodeDecodeError):
        return []


def reset_agent_memory(agent):
    """Reset/clear agent memory."""
    agent_file = _get_agent_file_path(agent.get_agent_id())
    
    try:
        if agent_file.exists():
            agent_file.unlink()
    except OSError:
        pass  # Silently fail if we can't delete the file


