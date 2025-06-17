import json
import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path


class MemoryManager:
    """Memory manager for storing and retrieving agent conversations in separate JSON files."""
    
    def __init__(self, memory_dir: Optional[str] = None):
        """
        Initialize the memory manager.
        
        Args:
            memory_dir: Directory to store individual agent memory files. 
                       If None, uses the same directory as this file.
        """
        if memory_dir is None:
            # Use the same directory as the current file
            current_file_dir = Path(__file__).parent
            self.memory_dir = current_file_dir
        else:
            self.memory_dir = Path(memory_dir)
        
        # Ensure directory exists
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_agent_file_path(self, agent_id: str) -> Path:
        """
        Generate a unique file path for an agent using SHA256 hash.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Path to the agent's memory file
        """
        # Create SHA256 hash of agent_id
        agent_hash = hashlib.sha256(agent_id.encode('utf-8')).hexdigest()
        return self.memory_dir / f"{agent_hash}.json"
    
    def _load_agent_memory(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Load memory data for a specific agent from their JSON file.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            List of messages for the agent, empty list if file doesn't exist or has issues
        """
        agent_file = self._get_agent_file_path(agent_id)
        
        try:
            if not agent_file.exists():
                return []
            
            with open(agent_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure we return a list
                if isinstance(data, list):
                    return data
                else:
                    return []
        except (json.JSONDecodeError, FileNotFoundError, OSError, UnicodeDecodeError):
            # Return empty list for any JSON or file operation errors
            return []
    
    def _save_agent_memory(self, agent_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Save memory data for a specific agent to their JSON file.
        
        Args:
            agent_id: Unique identifier for the agent
            messages: List of messages to save
        """
        agent_file = self._get_agent_file_path(agent_id)
        
        try:
            with open(agent_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except (OSError, UnicodeEncodeError):
            # Silently fail if we can't write to file
            pass
    
    def save_memory(self, agent_id: str, json_messages: List[Dict[str, Any]]) -> None:
        """
        Save messages for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            json_messages: List of message dictionaries to save
        """
        if not isinstance(json_messages, list):
            # Convert to empty list if not a list
            json_messages = []
        
        self._save_agent_memory(agent_id, json_messages)
    
    def get_memory(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve messages for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            List of message dictionaries for the agent, empty list if agent not found or errors
        """
        return self._load_agent_memory(agent_id)
    
    def reset_memory(self, agent_id: str) -> None:
        """
        Reset/clear messages for a specific agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        agent_file = self._get_agent_file_path(agent_id)
        
        try:
            if agent_file.exists():
                agent_file.unlink()
        except OSError:
            # Silently fail if we can't delete the file
            pass


# Create a default memory manager instance
_default_memory_manager = MemoryManager()


def save_memory(agent_id: str, json_messages: List[Dict[str, Any]]) -> None:
    """
    Save messages for a specific agent using the default memory manager.
    
    Args:
        agent_id: Unique identifier for the agent
        json_messages: List of message dictionaries to save
    """
    _default_memory_manager.save_memory(agent_id, json_messages)


def get_memory(agent_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve messages for a specific agent using the default memory manager.
    
    Args:
        agent_id: Unique identifier for the agent
        
    Returns:
        List of message dictionaries for the agent, empty list if agent not found or errors
    """
    return _default_memory_manager.get_memory(agent_id)


def reset_memory(agent_id: str) -> None:
    """
    Reset/clear messages for a specific agent using the default memory manager.
    
    Args:
        agent_id: Unique identifier for the agent
    """
    _default_memory_manager.reset_memory(agent_id)
