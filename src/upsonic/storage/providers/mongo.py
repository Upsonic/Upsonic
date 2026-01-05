from __future__ import annotations

import time
from typing import Optional, Type, Union, TypeVar, List, TYPE_CHECKING

if TYPE_CHECKING:
    from motor.motor_asyncio import (
        AsyncIOMotorClient,
        AsyncIOMotorDatabase,
        AsyncIOMotorCollection,
    )
    from upsonic.culture.cultural_knowledge import CulturalKnowledge

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
from upsonic.session.agent import AgentSession

T = TypeVar("T", bound=BaseModel)


class MongoStorage(Storage):
    """
    A high-performance, asynchronous storage provider for MongoDB, designed for
    scalability and idiomatic database interaction. It uses the `motor` driver,
    leverages native `_id` for primary keys, and ensures critical indexes
    for fast lookups.
    
    This storage provider is designed to be flexible and dynamic:
    - Can accept a pre-existing motor database or client, or create one from connection details
    - Only creates AgentSession collections/indexes when they are actually used
    - Supports generic Pydantic models for custom storage needs
    - Can be used for both custom purposes and built-in chat/profile features simultaneously
    """

    def __init__(
        self,
        database: Optional['AsyncIOMotorDatabase'] = None,
        client: Optional['AsyncIOMotorClient'] = None,
        db_url: Optional[str] = None,
        database_name: Optional[str] = None,
        sessions_collection_name: str = "agent_sessions",
        cultural_knowledge_collection_name: str = "cultural_knowledge",
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
            sessions_collection_name: The name of the collection for AgentSession.
                Only used if AgentSession objects are stored. Defaults to "agent_sessions".
            cultural_knowledge_collection_name: The name of the collection for CulturalKnowledge.
                Only used if CulturalKnowledge objects are stored. Defaults to "cultural_knowledge".
        """
        if not _MOTOR_AVAILABLE:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="motor",
                install_command='pip install "upsonic[storage]"',
                feature_name="MongoDB storage provider"
            )

        super().__init__()
        
        self._db: Optional[AsyncIOMotorDatabase] = database
        self._client: Optional[AsyncIOMotorClient] = client
        self._owns_client = (database is None and client is None)
        
        if not database and not client and not db_url:
            raise ValueError("Either 'database', 'client', or 'db_url' must be provided")
        if not database and not database_name:
            raise ValueError("'database_name' is required when 'database' is not provided")
        
        self.db_url = db_url
        self.database_name = database_name
        
        self.sessions_collection_name = sessions_collection_name
        self.cultural_knowledge_collection_name = cultural_knowledge_collection_name
        
        self._sessions_collection_initialized = False
        self._cultural_knowledge_collection_initialized = False



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

    def upsert(self, data: BaseModel) -> None:
        return self._run_async_from_sync(self.upsert_async(data))

    def delete(self, object_id: str, model_type: Type[BaseModel]) -> None:
        return self._run_async_from_sync(self.delete_async(object_id, model_type))

    def drop(self) -> None:
        return self._run_async_from_sync(self.drop_async())



    async def connect_async(self) -> None:
        if await self.is_connected_async():
            return
        
        if not self._owns_client:
            if self._db:
                self._connected = True
                return
            elif self._client:
                self._db = self._client[self.database_name]
                self._connected = True
                return
        
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
        if not self._owns_client:
            return
        
        if self._client:
            self._client.close()
        self._client = None
        self._db = None
        self._connected = False

    async def is_connected_async(self) -> bool:
        return self._client is not None and self._db is not None

    async def create_async(self) -> None:
        if not await self.is_connected_async():
            await self.connect_async()
    
    async def _ensure_sessions_collection(self) -> None:
        if self._sessions_collection_initialized:
            return
        
        if self._db is None:
            raise ConnectionError(
                "Cannot create indexes without a database connection. Call connect() first."
            )
        
        sessions_collection = self._db[self.sessions_collection_name]
        await sessions_collection.create_index("user_id")
        self._sessions_collection_initialized = True
    
    async def _ensure_cultural_knowledge_collection(self) -> None:
        if self._cultural_knowledge_collection_initialized:
            return
        
        if self._db is None:
            raise ConnectionError(
                "Cannot create collection without a database connection. Call connect() first."
            )
        
        collection = self._db[self.cultural_knowledge_collection_name]
        await collection.create_index("name")
        self._cultural_knowledge_collection_initialized = True

    async def read_async(self, object_id: str, model_type: Type[T]) -> Optional[T]:
        import base64
        if model_type.__name__ == "AgentSession":
            await self._ensure_sessions_collection()
        
        collection = await self._get_collection_for_model_async(model_type)
        id_field_name = self._get_id_field(model_type)
        doc = await collection.find_one({"_id": object_id})
        if doc:
            if model_type.__name__ == "AgentSession" and "data" in doc:
                # Use deserialize: base64 decode then deserialize
                serialized_bytes = base64.b64decode(doc["data"].encode('utf-8'))
                return model_type.deserialize(serialized_bytes)
            else:
                doc[id_field_name] = doc.pop("_id")
                if hasattr(model_type, 'from_dict'):
                    return model_type.from_dict(doc)
                return model_type.model_validate(doc)
        return None

    async def upsert_async(self, data: BaseModel) -> None:
        import base64
        if type(data).__name__ == "AgentSession":
            await self._ensure_sessions_collection()
        
        collection = await self._get_collection_for_model_async(type(data))
        id_field_name = self._get_id_field(data)
        object_id = getattr(data, id_field_name)
        
        if hasattr(data, 'updated_at'):
            data.updated_at = time.time()
        
        if type(data).__name__ == "AgentSession":
            # Use serialize: serialize to bytes, then base64 encode
            serialized_bytes = data.serialize()
            serialized_str = base64.b64encode(serialized_bytes).decode('utf-8')
            doc = {
                "_id": object_id,
                "data": serialized_str,
                "created_at": data.created_at or time.time(),
                "updated_at": data.updated_at or time.time()
            }
        elif hasattr(data, 'to_dict'):
            doc = data.to_dict()
            doc["_id"] = doc.pop(id_field_name)
        else:
            doc = data.model_dump()
            doc["_id"] = doc.pop(id_field_name)
        await collection.replace_one({"_id": object_id}, doc, upsert=True)

    async def delete_async(self, object_id: str, model_type: Type[BaseModel]) -> None:
        if model_type.__name__ == "AgentSession":
            await self._ensure_sessions_collection()
        
        collection = await self._get_collection_for_model_async(model_type)
        await collection.delete_one({"_id": object_id})
    
    async def list_all_async(self, model_type: Type[T]) -> List[T]:
        import base64
        try:
            if model_type.__name__ == "AgentSession":
                await self._ensure_sessions_collection()
            
            collection = await self._get_collection_for_model_async(model_type)
            id_field_name = self._get_id_field(model_type)
            
            results = []
            cursor = collection.find({})
            
            async for doc in cursor:
                try:
                    if model_type.__name__ == "AgentSession" and "data" in doc:
                        # Use deserialize: base64 decode then deserialize
                        serialized_bytes = base64.b64decode(doc["data"].encode('utf-8'))
                        obj = model_type.deserialize(serialized_bytes)
                    else:
                        doc[id_field_name] = doc.pop("_id")
                        if hasattr(model_type, 'from_dict'):
                            obj = model_type.from_dict(doc)
                        else:
                            obj = model_type.model_validate(doc)
                    results.append(obj)
                except Exception:
                    continue
            
            return results
        except Exception:
            return []

    async def drop_async(self) -> None:
        if self._db is None:
            return
        
        try:
            await self._db.drop_collection(self.sessions_collection_name)
        except Exception:
            pass
        try:
            await self._db.drop_collection(self.cultural_knowledge_collection_name)
        except Exception:
            pass
        
        self._sessions_collection_initialized = False
        self._cultural_knowledge_collection_initialized = False



    async def _get_collection_for_model_async(
        self, model_type: Type[BaseModel]
    ) -> AsyncIOMotorCollection:
        """Get collection with auto-reconnect if connection is lost."""
        await self._ensure_connection()
        
        if model_type.__name__ == "AgentSession":
            return self._db[self.sessions_collection_name]
        else:
            collection_name = f"{model_type.__name__.lower()}_storage"
            return self._db[collection_name]
    
    async def _ensure_connection(self) -> None:
        """Ensure database connection is valid, reconnect if needed."""
        needs_reconnect = False
        
        if self._db is None:
            needs_reconnect = True
        elif self._client is not None:
            # Check if connection is still valid with a ping
            try:
                await self._client.admin.command("ping")
            except Exception:
                needs_reconnect = True
        
        if needs_reconnect:
            # Close existing connection if it exists
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
                self._db = None
            
            # Reconnect
            await self.connect_async()
    
    def _get_collection_for_model(
        self, model_type: Type[BaseModel]
    ) -> AsyncIOMotorCollection:
        """Get collection (sync version - does not auto-reconnect)."""
        if self._db is None:
            raise ConnectionError(
                "Not connected to the database. Call connect() or connect_async() first."
            )
        if model_type.__name__ == "AgentSession":
            return self._db[self.sessions_collection_name]
        else:
            collection_name = f"{model_type.__name__.lower()}_storage"
            return self._db[collection_name]

    @staticmethod
    def _get_id_field(model_or_type: Union[BaseModel, Type[BaseModel]]) -> str:
        model_type = (
            model_or_type if isinstance(model_or_type, type) else type(model_or_type)
        )
        if model_type.__name__ == "AgentSession":
            return "session_id"
        else:
            if hasattr(model_type, 'model_fields'):
                for field_name in ['path', 'id', 'key', 'name']:
                    if field_name in model_type.model_fields:
                        return field_name
            return "id"

    # =========================================================================
    # Cultural Knowledge Methods
    # =========================================================================

    async def read_cultural_knowledge_async(self, knowledge_id: str) -> Optional["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        await self._ensure_cultural_knowledge_collection()
        
        collection = self._db[self.cultural_knowledge_collection_name]
        doc = await collection.find_one({"_id": knowledge_id})
        
        if doc:
            doc["id"] = doc.pop("_id")
            return CulturalKnowledge.from_dict(doc)
        return None

    async def upsert_cultural_knowledge_async(self, knowledge: "CulturalKnowledge") -> None:
        await self._ensure_cultural_knowledge_collection()
        
        knowledge.bump_updated_at()
        
        collection = self._db[self.cultural_knowledge_collection_name]
        
        doc = knowledge.to_dict()
        doc["_id"] = doc.pop("id")
        
        await collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    async def delete_cultural_knowledge_async(self, knowledge_id: str) -> None:
        await self._ensure_cultural_knowledge_collection()
        
        collection = self._db[self.cultural_knowledge_collection_name]
        await collection.delete_one({"_id": knowledge_id})

    async def list_all_cultural_knowledge_async(
        self, 
        name: Optional[str] = None
    ) -> List["CulturalKnowledge"]:
        from upsonic.culture.cultural_knowledge import CulturalKnowledge
        
        try:
            await self._ensure_cultural_knowledge_collection()
            
            collection = self._db[self.cultural_knowledge_collection_name]
            
            if name:
                query = {"name": {"$regex": name, "$options": "i"}}
            else:
                query = {}
            
            results = []
            cursor = collection.find(query)
            
            async for doc in cursor:
                try:
                    doc["id"] = doc.pop("_id")
                    results.append(CulturalKnowledge.from_dict(doc))
                except Exception:
                    continue
            
            return results
        except Exception:
            return []

    async def clear_cultural_knowledge_async(self) -> None:
        await self._ensure_cultural_knowledge_collection()
        
        collection = self._db[self.cultural_knowledge_collection_name]
        await collection.delete_many({})
