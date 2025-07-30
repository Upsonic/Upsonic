from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from upsonic.storage.session import BaseSession


class Artifact(BaseModel):
    """
    A placeholder for the Artifact metadata model.
    Represents a non-textual file associated with a session.
    """
    artifact_id: str = Field(..., description="A unique identifier for the artifact.")
    session_id: str = Field(..., description="Foreign key linking this artifact to a specific session.")
    storage_uri: str = Field(..., description="The URI pointing to the artifact's actual location in a blob store.")


class SchemaMismatchError(Exception):
    """
    Custom exception raised when the database schema does not match the
    application's expected schema.
    """
    pass



class Storage(ABC):
    """
    The "Contract" for a Session-Based Interaction Archive.

    This Abstract Base Class defines the universal API for persisting and
    retrieving agent interaction sessions. The contract is designed around a
    stateful session model, where an entire session's state (including its
    message history) is updated at once via an 'upsert' operation.

    This replaces the previous append-only 'LLMConversation' model with a more
    flexible and robust state management pattern.
    """

    def __init__(self, mode: Optional[Literal["agent", "team", "workflow", "workflow_v2"]] = "agent"):
        self._mode = mode
        self._connected = False

    @property
    def mode(self) -> Optional[Literal["agent", "team", "workflow", "workflow_v2"]]:
        """Get the mode or namespace of the storage instance."""
        return self._mode

    @mode.setter
    def mode(self, value: Optional[Literal["agent", "team", "workflow", "workflow_v2"]]) -> None:
        """Set the mode of the storage."""
        self._mode = value

    @abstractmethod
    def is_connected(self) -> bool:
        """Checks if the storage provider is currently connected."""
        raise NotImplementedError

    @abstractmethod
    def connect(self) -> None:
        """
        Establishes and verifies the connection to the storage backend.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Closes the connection to the storage backend gracefully."""
        raise NotImplementedError

    @abstractmethod
    def create(self) -> None:
        """
        Ensures the required storage infrastructure (e.g., tables, collections)
        exists. Should be safe to call multiple times.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, session_id: str, user_id: Optional[str] = None) -> Optional[BaseSession]:
        """
        Reads a single, complete Session from the storage.

        Args:
            session_id: The unique ID of the session to retrieve.
            user_id: An optional user ID for an additional layer of access control.

        Returns:
            A Session object if found, otherwise None.
        """
        raise NotImplementedError

    @abstractmethod
    def upsert(self, session: BaseSession) -> Optional[BaseSession]:
        """
        Inserts a new Session or updates an existing one.

        This is the primary write method. If a session with the given
        `session.session_id` exists, it will be completely updated with the
        new session data. If it does not exist, it will be created. This
        operation should be atomic.

        Args:
            session: The Session object containing the full state to save.

        Returns:
            The upserted Session object, reflecting the state after the
            operation, or None if the operation failed.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None) -> List[BaseSession]:
        """
        Retrieves all sessions, optionally filtered.

        Args:
            user_id: The ID of the user to filter by.
            entity_id: The ID of the agent, team, or workflow to filter by.

        Returns:
            A list of Session objects. For performance, implementations may
            choose to exclude heavyweight fields like the full memory history.
        """
        raise NotImplementedError

    @abstractmethod
    def get_recent_sessions(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 10) -> List[BaseSession]:
        """
        Retrieves the N most recently updated sessions.

        Args:
            user_id: The ID of the user to filter by.
            entity_id: The ID of the agent, team, or workflow to filter by.
            limit: The maximum number of recent sessions to return.

        Returns:
            A list of the most recent Session objects.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """
        Deletes a session and all its associated data from the storage.

        Args:
            session_id: The ID of the session to delete.
        """
        raise NotImplementedError

    @abstractmethod
    def drop(self) -> None:
        """
        Deletes ALL data from the storage provider corresponding to the current
        configuration. This is a destructive operation intended for testing
        and development.
        """
        raise NotImplementedError


    @abstractmethod
    def log_artifact(self, artifact: Artifact) -> None:
        """
        Records the metadata of an artifact associated with a session.

        Args:
            artifact: The Artifact metadata object to save.
        """
        raise NotImplementedError

    @abstractmethod
    def store_artifact_data(self, artifact_id: str, session_id: str, binary_data: bytes) -> str:
        """
        Stores the raw binary data of an artifact and returns its access URI.

        Args:
            artifact_id: The unique ID of the artifact being stored.
            session_id: The ID of the session for namespacing/organization.
            binary_data: The raw bytes of the file to store.

        Returns:
            A unique storage URI (e.g., "file:///path/to/file") that the
            provider can use later to retrieve the data.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        """
        Retrieves the raw binary data of an artifact using its storage URI.

        Args:
            storage_uri: The unique storage URI generated by the same provider.

        Returns:
            The raw bytes of the artifact.

        Raises:
            FileNotFoundError: If no artifact data can be found for the given URI.
        """
        raise NotImplementedError