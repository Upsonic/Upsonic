from __future__ import annotations

import time
from typing import Optional, Type, Union, TypeVar, List, TYPE_CHECKING

if TYPE_CHECKING:
    from motor.motor_asyncio import (
        AsyncIOMotorClient,
        AsyncIOMotorDatabase,
        AsyncIOMotorCollection,
    )

try:
    from motor.motor_asyncio import (
        AsyncIOMotorClient,
        AsyncIOMotorDatabase,
        AsyncIOMotorCollection,
    )
    _MOTOR_AVAILABLE = True
except ImportError:
    AsyncIOMotorClient = None  # type: ignore
    AsyncIOMotorDatabase = None  # type: ignore
    AsyncIOMotorCollection = None  # type: ignore
    _MOTOR_AVAILABLE = False

from pydantic import BaseModel

from upsonic.storage.base import Storage
from upsonic.storage.session.sessions import InteractionSession, UserProfile

T = TypeVar("T", bound=BaseModel)


class MongoStorage(Storage):
    """
    A high-performance, asynchronous storage provider for MongoDB, designed for
    scalability and idiomatic database interaction. It uses the `motor` driver,
    leverages native `_id` for primary keys, and ensures critical indexes
    for fast lookups.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing motor database or client, or create one from connection details
    - Only creates InteractionSession/UserProfile collections/indexes when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        database: Optional['AsyncIOMotorDatabase'] = None,
        client: Optional['AsyncIOMotorClient'] = None,
        db_url: Optional[str] = None,
        database_name: Optional[str] = None,
        sessions_collection_name: str = "interaction_sessions",
        profiles_collection_name: str = "user_profiles",
    ):
        """
        Initializes the async MongoDB storage provider.

        Args:
            database: Optional pre-existing AsyncIOMotorDatabase. If provided, this database
                will be used. User is responsible for database lifecycle management.
            client: Optional pre-existing AsyncIOMotorClient. If provided and database is not,
                database_name will be used to get the database from this client.
            db_url: The full MongoDB connection string (e.g., "mongodb://localhost:27017").
                Required if database and client are not provided.
            database_name: The name of the database to use. Required if database is not provided.
            sessions_collection_name: The name of the collection for InteractionSession.
                Only used if InteractionSession objects are stored. Defaults to "interaction_sessions".
            profiles_collection_name: The name of the collection for UserProfile.
                Only used if UserProfile objects are stored. Defaults to "user_profiles".
        """
        if not _MOTOR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="motor",
                install_command='pip install "upsonic[storage]"',
                feature_name="MongoDB storage provider"
            )

        super().__init__()
        
        # Store database/client and track ownership for lifecycle management
        self._db: Optional[AsyncIOMotorDatabase] = database
        self._client: Optional[AsyncIOMotorClient] = client
        self._owns_client = (database is None and client is None)  # True if we create it
        
        # Connection details for creating our own client/db if needed
        if not database and not client and not db_url:
            raise ValueError("Either 'database', 'client', or 'db_url' must be provided")
        if not database and not database_name:
            raise ValueError("'database_name' is required when 'database' is not provided")
        
        self.db_url = db_url
        self.database_name = database_name
        
        # Collection names for InteractionSession/UserProfile (lazy initialization)
        self.sessions_collection_name = sessions_collection_name
        self.profiles_collection_name = profiles_collection_name
        
        # Track which built-in collections have been initialized
        self._sessions_collection_initialized = False
        self._profiles_collection_initialized = False



    def is_connected(self) -> bool:
        return self._run_async_from_sync(self.is_connected_async())

    def connect(self) -> None:
        return self._run_async_from_sync(self.connect_async())

    def disconnect(self) -> None:
        return self._run_async_from_sync(self.disconnect_async())

    def create(self) -> None:
        return self._run_async_from_sync(self.create_async())

    def read(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        return self._run_async_from_sync(self.read_async(object_id, model_type))

    def upsert(self, data: Union[InteractionSession, UserProfile]) -> None:
        return self._run_async_from_sync(self.upsert_async(data))

    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))

    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())



    async def connect_async(self) -> None:
        """
        Establishes connection to the database.
        If user provided a database or client, this is a no-op or minimal setup.
        Otherwise, creates a new client and database.
        """
        if await self.is_connected_async():
            return
        
        if not self._owns_client:
            # User provided database or client
            if self._db:
                # Database already set
                self._connected = True
                return
            elif self._client:
                # Client provided, get database from it
                self._db = self._client[self.database_name]
                self._connected = True
                return
        
        # Create our own client and database
        try:
            self._client = AsyncIOMotorClient(self.db_url)
            await self._client.admin.command("ismaster")
            self._db = self._client[self.database_name]
            self._connected = True
        except Exception as e:
            self._client = None
            self._db = None
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to MongoDB at {self.db_url}: {e}"
            ) from e

    async def disconnect_async(self) -> None:
        """
        Closes the database connection.
        If user provided the database or client, this is a no-op (user manages lifecycle).
        """
        if not self._owns_client:
            # User manages their own database/client lifecycle
            return
        
        if self._client:
            self._client.close()
        self._client = None
        self._db = None
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._client is not None and self._db is not None

    async def create_async(self) -> None:
        """
        Creates database schema.
        Note: InteractionSession and UserProfile collections/indexes are created lazily
        only when first accessed. This allows the storage to be used for
        generic purposes without creating unused infrastructure.
        """
        # Ensure connection exists, but don't create any collections/indexes yet
        # Collections/indexes will be created on-demand when accessed
        if not await self.is_connected_async():
            await self.connect_async()
    
    async def _ensure_sessions_collection(self) -> None:
        """Lazily creates the sessions collection and indexes on first access."""
        if self._sessions_collection_initialized:
            return
        
        if self._db is None:
            raise ConnectionError(
                "Cannot create indexes without a database connection. Call connect() first."
            )
        
        sessions_collection = self._db[self.sessions_collection_name]
        await sessions_collection.create_index("user_id")
        self._sessions_collection_initialized = True
    
    async def _ensure_profiles_collection(self) -> None:
        """Lazily creates the profiles collection on first access."""
        if self._profiles_collection_initialized:
            return
        
        if self._db is None:
            raise ConnectionError(
                "Cannot create collection without a database connection. Call connect() first."
            )
        
        # Profiles collection doesn't need any special indexes beyond _id
        # Just mark as initialized
        self._profiles_collection_initialized = True

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        # Ensure collection exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_collection()
        elif model_type is UserProfile:
            await self._ensure_profiles_collection()
        # Generic models don't need explicit initialization
        
        collection = self._get_collection_for_model(model_type)
        id_field_name = self._get_id_field(model_type)
        doc = await collection.find_one({"_id": object_id})
        if doc:
            doc[id_field_name] = doc.pop("_id")
            return model_type.model_validate(doc)
        return None

    async def upsert_async(self, data: BaseModel) -> None:
        # Ensure collection exists - lazy initialization for built-in types
        if isinstance(data, InteractionSession):
            await self._ensure_sessions_collection()
        elif isinstance(data, UserProfile):
            await self._ensure_profiles_collection()
        # Generic models don't need explicit initialization
        
        collection = self._get_collection_for_model(type(data))
        id_field_name = self._get_id_field(data)
        object_id = getattr(data, id_field_name)
        
        # Update timestamp if field exists
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        doc = data.model_dump()
        doc["_id"] = doc.pop(id_field_name)
        await collection.replace_one({"_id": object_id}, doc, upsert=True)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        # Ensure collection exists - lazy initialization for built-in types
        if model_type is InteractionSession:
            await self._ensure_sessions_collection()
        elif model_type is UserProfile:
            await self._ensure_profiles_collection()
        # For generic models, only delete if collection exists
        
        collection = self._get_collection_for_model(model_type)
        await collection.delete_one({"_id": object_id})
    
    async def list_all_async(self, model_type: Type[T]) -> List[T]:
        """List all objects of a specific type."""
        try:
            # Ensure collection exists - lazy initialization for built-in types
            if model_type is InteractionSession:
                await self._ensure_sessions_collection()
            elif model_type is UserProfile:
                await self._ensure_profiles_collection()
            # Generic models don't need explicit initialization
            
            collection = self._get_collection_for_model(model_type)
            id_field_name = self._get_id_field(model_type)
            
            results = []
            cursor = collection.find({})
            
            async for doc in cursor:
                try:
                    # Convert MongoDB's _id back to model's ID field
                    doc[id_field_name] = doc.pop("_id")
                    obj = model_type.model_validate(doc)
                    results.append(obj)
                except Exception:
                    continue
            
            return results
        except Exception:
            return []

    async def drop_async(self) -> None:
        """
        Drops all collections managed by this storage provider.
        Only drops InteractionSession/UserProfile collections if they were actually created.
        """
        if self._db is None:
            return
        
        try:
            await self._db.drop_collection(self.sessions_collection_name)
        except Exception:
            pass
        try:
            await self._db.drop_collection(self.profiles_collection_name)
        except Exception:
            pass
        
        # Reset initialization flags
        self._sessions_collection_initialized = False
        self._profiles_collection_initialized = False

    async def read_sessions_for_user_async(self, user_id: str) -> List[InteractionSession]:
        """
        Retrieves all interaction sessions associated with a specific user ID,
        leveraging the secondary index on the `user_id` field for high performance.

        Args:
            user_id: The ID of the user whose sessions are to be retrieved.

        Returns:
            A list of InteractionSession objects, which may be empty if the user
            has no sessions.
        """
        # Ensure sessions collection exists
        await self._ensure_sessions_collection()
        
        collection = self._get_collection_for_model(InteractionSession)
        cursor = collection.find({"user_id": user_id})
        sessions = []
        id_field_name = self._get_id_field(InteractionSession)
        
        async for doc in cursor:
            doc[id_field_name] = doc.pop("_id")
            sessions.append(InteractionSession.model_validate(doc))
            
        return sessions



    def _get_collection_for_model(
        self, model_type: Type[BaseModel]
    ) -> AsyncIOMotorCollection:
        if self._db is None:
            raise ConnectionError(
                "Not connected to the database. Call connect() or connect_async() first."
            )
        if model_type is InteractionSession:
            return self._db[self.sessions_collection_name]
        elif model_type is UserProfile:
            return self._db[self.profiles_collection_name]
        else:
            # Generic model support: collection name is {model_name}_storage
            collection_name = f"{model_type.__name__.lower()}_storage"
            return self._db[collection_name]

    @staticmethod
    def _get_id_field(model_or_type: Union[BaseModel, Type[BaseModel]]) -> str:
        model_type = (
            model_or_type if isinstance(model_or_type, type) else type(model_or_type)
        )
        if model_type is InteractionSession:
            return "session_id"
        elif model_type is UserProfile:
            return "user_id"
        else:
            # Generic model: auto-detect primary key
            if hasattr(model_type, 'model_fields'):
                for field_name in ['path', 'id', 'key', 'name']:
                    if field_name in model_type.model_fields:
                        return field_name
            return "id"