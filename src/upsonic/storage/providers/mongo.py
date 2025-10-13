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
    """

    def __init__(
        self,
        db_url: str,
        database_name: str,
        sessions_collection_name: str = "interaction_sessions",
        profiles_collection_name: str = "user_profiles",
    ):
        """
        Initializes the async MongoDB storage provider.

        Args:
            db_url: The full MongoDB connection string (e.g., "mongodb://localhost:27017").
            database_name: The name of the database to use.
            sessions_collection_name: The name of the collection for InteractionSession.
            profiles_collection_name: The name of the collection for UserProfile.
        """
        if not _MOTOR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="motor",
                install_command='pip install "upsonic[storage]"',
                feature_name="MongoDB storage provider"
            )

        super().__init__()
        self.db_url = db_url
        self.database_name = database_name
        self.sessions_collection_name = sessions_collection_name
        self.profiles_collection_name = profiles_collection_name

        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None



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
        if await self.is_connected_async():
            return
        try:
            self._client = AsyncIOMotorClient(self.db_url)
            await self._client.admin.command("ismaster")
            self._db = self._client[self.database_name]
            await self.create_async()
            self._connected = True
        except Exception as e:
            self._client = None
            self._db = None
            self._connected = False
            raise ConnectionError(
                f"Failed to connect to MongoDB at {self.db_url}: {e}"
            ) from e

    async def disconnect_async(self) -> None:
        if self._client:
            self._client.close()
        self._client = None
        self._db = None
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._client is not None and self._db is not None

    async def create_async(self) -> None:
        if self._db is None:
            raise ConnectionError(
                "Cannot create indexes without a database connection. Call connect() first."
            )
        sessions_collection = self._db[self.sessions_collection_name]
        await sessions_collection.create_index("user_id")

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        collection = self._get_collection_for_model(model_type)
        id_field_name = self._get_id_field(model_type)
        doc = await collection.find_one({"_id": object_id})
        if doc:
            doc[id_field_name] = doc.pop("_id")
            return model_type.model_validate(doc)
        return None

    async def upsert_async(self, data: Union[InteractionSession, UserProfile]) -> None:
        collection = self._get_collection_for_model(type(data))
        id_field_name = self._get_id_field(data)
        object_id = getattr(data, id_field_name)
        data.updated_at = time.time()
        doc = data.model_dump()
        doc["_id"] = doc.pop(id_field_name)
        await collection.replace_one({"_id": object_id}, doc, upsert=True)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        collection = self._get_collection_for_model(model_type)
        await collection.delete_one({"_id": object_id})

    async def drop_async(self) -> None:
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
            raise TypeError(
                f"Unsupported model type for MongoDB storage: {model_type.__name__}"
            )

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
            raise TypeError(f"Unsupported model type: {model_or_type}")