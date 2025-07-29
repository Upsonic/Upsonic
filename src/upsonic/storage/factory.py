from typing import Any, Dict, Type

from upsonic.storage.base import Storage
from upsonic.storage.providers.in_memory import InMemoryStorage
from upsonic.storage.providers.json import JSONStorage
from upsonic.storage.providers.postgres import PostgresStorage
from upsonic.storage.providers.redis import RedisStorage
from upsonic.storage.providers.sqlite import SqliteStorage
from upsonic.storage.settings import (
    StorageSettings,
    PostgresSettings,
    SQLiteSettings,
    RedisSettings,
    JSONSettings,
    InMemorySettings,
)

_PROVIDER_CLASS_MAP: Dict[str, Type[Storage]] = {
    "postgres": PostgresStorage,
    "sqlite": SqliteStorage,
    "redis": RedisStorage,
    "json": JSONStorage,
    "in_memory": InMemoryStorage,
}

_SETTINGS_CLASS_MAP: Dict[str, Type[StorageSettings]] = {
    "postgres": PostgresSettings,
    "sqlite": SQLiteSettings,
    "redis": RedisSettings,
    "json": JSONSettings,
    "in_memory": InMemorySettings,
}


class StorageFactory:
    """
    An intelligent factory for creating storage provider instances.

    This factory orchestrates the entire storage initialization process:
    1. It reads the base configuration to determine the storage type.
    2. It loads and validates the specific settings for that type.
    3. It instantiates and returns the correct storage provider.

    The application code interacts with this factory to get a storage
    instance, fully abstracting away the configuration details.
    """

    @staticmethod
    def get_storage() -> Storage:
        """
        Instantiates and returns a storage provider based on environment variables.

        This function is the single entry point for the storage system. It reads
        the .env file and system environment to determine which provider to use
        and how to configure it.

        The process follows a robust two-step validation:
        1.  Validate `STORAGE_TYPE` to decide which provider to use.
        2.  Validate the specific settings required for that provider (e.g.,
            `POSTGRES_DB_URI` for postgres).

        Example Usage:
        
        # In your application's main file:
        # (Assuming .env file is configured)
        > storage = StorageFactory.get_storage()
        > Agent(storage=storage)

        Returns:
            An instance of a class that implements the Storage contract,
            fully configured and ready to be connected.

        Raises:
            ValueError: If the configuration is invalid (e.g., missing required
                        variables, unknown storage type, or typos in variable names).
        """
        try:
            base_settings = StorageSettings()
            storage_type = base_settings.STORAGE_TYPE

            SettingsClass = _SETTINGS_CLASS_MAP.get(storage_type)
            if not SettingsClass:
                raise ValueError(f"Internal factory error: No settings class found for type '{storage_type}'.")
            
            specialized_settings = SettingsClass()

            ProviderClass = _PROVIDER_CLASS_MAP.get(storage_type)
            if not ProviderClass:
                raise ValueError(f"Internal factory error: No provider class found for type '{storage_type}'.")
            
            storage_instance = ProviderClass(settings=specialized_settings)

            return storage_instance

        except Exception as e:
            raise ValueError(f"Failed to initialize storage provider due to a configuration error: {e}") from e


get_storage = StorageFactory.get_storage