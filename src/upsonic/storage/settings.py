from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, PostgresDsn, RedisDsn, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """
    The base settings class.

    Its primary role is to read the STORAGE_TYPE from the environment to determine
    which specialized settings class should be used.
    """
    STORAGE_TYPE: Literal["postgres", "sqlite", "redis", "json", "in_memory"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class PostgresSettings(StorageSettings):
    """Settings required for the PostgresStorage provider."""

    POSTGRES_DB_URI: PostgresDsn

    POSTGRES_TABLE_NAME: str = Field(default="agent_sessions", description="The name of the table for session storage.")
    
    POSTGRES_SCHEMA: str = Field(default="public", description="The PostgreSQL schema to use for the session table.")

    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="ignore")


class SQLiteSettings(StorageSettings):
    """Settings required for the SQLiteStorage provider."""
    SQLITE_DB_PATH: Path = Field(default="upsonic_storage.db", description="Path to the SQLite database file.")
    SQLITE_TABLE_NAME: str = Field(default="agent_sessions", description="The name of the table to store sessions.")
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="ignore")


class RedisSettings(StorageSettings):
    """Settings required for the RedisStorage provider."""

    REDIS_DSN: RedisDsn = "redis://localhost:6379/0"

    REDIS_PREFIX: str

    REDIS_EXPIRE: Optional[int] = None

    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="ignore")

    @model_validator(mode='after')
    def handle_ssl_in_dsn(self) -> 'RedisSettings':
        """Automatically adjusts the DSN scheme for SSL if needed."""
        dsn_str = str(self.REDIS_DSN)
        if 'ssl=true' in dsn_str.lower() and dsn_str.startswith('redis://'):
            self.REDIS_DSN = RedisDsn(dsn_str.replace('redis://', 'rediss://', 1))
        return self


class JSONSettings(StorageSettings):
    """Settings required for the JSONStorage provider."""
    JSON_DIRECTORY_PATH: Path = Field(default="storage_data/")
    JSON_PRETTY_PRINT: bool = True
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="ignore")


class InMemorySettings(StorageSettings):
    """Settings for the InMemoryStorage provider."""
    IN_MEMORY_MAX_SESSIONS: Optional[int] = None
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="ignore")