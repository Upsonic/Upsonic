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
    POSTGRES_ARTIFACT_PATH: Path = Field(default="postgres_artifacts/")

    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="forbid")


class SQLiteSettings(StorageSettings):
    """Settings required for the SQLiteStorage provider."""
    # The default value provides a sensible default for local development.
    SQLITE_DB_PATH: Path = Field(default="upsonic_storage.db")
    SQLITE_ARTIFACT_PATH: Path = Field(default="sqlite_artifacts/")
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="forbid")


class RedisSettings(StorageSettings):
    """Settings required for the RedisStorage provider."""
 
    REDIS_DSN: RedisDsn = "redis://localhost:6379/0"

    REDIS_PREFIX: str

    REDIS_EXPIRE: Optional[int] = None

    REDIS_SSL: bool = False

    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="forbid")

    @model_validator(mode='after')
    def handle_ssl_in_dsn(self) -> 'RedisSettings':
        """Automatically adjusts the DSN scheme for SSL."""
        dsn_str = str(self.REDIS_DSN)
        
        if self.REDIS_SSL and dsn_str.startswith('redis://'):
            self.REDIS_DSN = RedisDsn(dsn_str.replace('redis://', 'rediss://', 1))
        elif not self.REDIS_SSL and dsn_str.startswith('rediss://'):
            self.REDIS_DSN = RedisDsn(dsn_str.replace('rediss://', 'redis://', 1))
            
        return self


class JSONSettings(StorageSettings):
    """Settings required for the JSONStorage provider."""
    JSON_DIRECTORY_PATH: Path = Field(default="storage_data/")
    JSON_ARTIFACT_PATH: Path = Field(default="storage_artifacts/")
    JSON_PRETTY_PRINT: bool = True
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="forbid")


class InMemorySettings(StorageSettings):
    """Settings for the InMemoryStorage provider."""
    IN_MEMORY_MAX_SESSIONS: Optional[int] = None
    STORAGE_MODE: Literal["agent", "team", "workflow", "workflow_v2"] = "agent"

    model_config = SettingsConfigDict(extra="forbid")