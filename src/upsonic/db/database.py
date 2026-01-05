from __future__ import annotations
from typing import Optional, Type, Union, Dict, Any, List, Literal, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    import aiosqlite
    from mem0 import Memory as Mem0Memory, AsyncMemoryClient as Mem0AsyncMemoryClient
    from ..storage.providers.mem0 import Mem0Storage
    from ..storage.providers.postgres import PostgresStorage
    from ..storage.providers.redis import RedisStorage
    from ..storage.providers.sqlite import SqliteStorage
    from ..storage.providers.mongo import MongoStorage
    from ..storage.providers.in_memory import InMemoryStorage
    from ..storage.providers.json import JSONStorage
    from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorClient
    from redis.asyncio import Redis

from ..storage.base import Storage
from ..storage.memory.memory import Memory
from ..models import Model
from ..storage.providers import (
    InMemoryStorage,
    JSONStorage,
    Mem0Storage,
    PostgresStorage,
    RedisStorage,
    SqliteStorage,
    MongoStorage
)


StorageType = TypeVar('StorageType', bound=Storage)


class DatabaseBase(Generic[StorageType]):
    """
    Base class for all database classes that combine storage providers with memory.
    """
    
    def __init__(
        self,
        storage: StorageType,
        memory: Memory
    ):
        self.storage = storage
        self.memory = memory
    
    @property
    def session_id(self) -> Optional[str]:
        """Get session_id from memory."""
        return self.memory.session_id if self.memory else None
    
    @property
    def user_id(self) -> Optional[str]:
        """Get user_id from memory."""
        return self.memory.user_id if self.memory else None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(storage={type(self.storage).__name__}, memory={type(self.memory).__name__})"


class SqliteDatabase(DatabaseBase[SqliteStorage]):
    """
    Database class combining SqliteStorage and Memory attributes.
    """
    
    def __init__(
        self,
        db: Optional[aiosqlite.Connection] = None,
        db_file: Optional[str] = None,
        agent_sessions_table_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = SqliteStorage(
            db=db,
            db_file=db_file,
            agent_sessions_table_name=agent_sessions_table_name
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class PostgresDatabase(DatabaseBase[PostgresStorage]):
    """
    Database class combining PostgresStorage and Memory attributes.
    """
    
    def __init__(
        self,
        pool: Optional[aiosqlite.Pool] = None,
        db_url: Optional[str] = None,
        schema: str = "public",
        agent_sessions_table_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = PostgresStorage(
            pool=pool,
            db_url=db_url,
            schema=schema,
            agent_sessions_table_name=agent_sessions_table_name
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class MongoDatabase(DatabaseBase[MongoStorage]):
    """
    Database class combining MongoStorage and Memory attributes.
    """
    
    def __init__(
        self,
        database: Optional['AsyncIOMotorDatabase'] = None,
        client: Optional['AsyncIOMotorClient'] = None,
        db_url: Optional[str] = None,
        database_name: Optional[str] = None,
        agent_sessions_collection_name: Optional[str] = None,
        cultural_knowledge_collection_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = MongoStorage(
            database=database,
            client=client,
            db_url=db_url,
            database_name=database_name or "upsonic",
            sessions_collection_name=agent_sessions_collection_name
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class RedisDatabase(DatabaseBase[RedisStorage]):
    """
    Database class combining RedisStorage and Memory attributes.
    """
    
    def __init__(
        self,
        redis_client: Optional['Redis'] = None,
        prefix: Optional[str] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        expire: Optional[int] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = RedisStorage(
            redis_client=redis_client,
            prefix=prefix,
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            expire=expire
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class InMemoryDatabase(DatabaseBase[InMemoryStorage]):
    """
    Database class combining InMemoryStorage and Memory attributes.
    """
    
    def __init__(
        self,
        max_sessions: Optional[int] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = InMemoryStorage(
            max_sessions=max_sessions
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class JSONDatabase(DatabaseBase[JSONStorage]):
    """
    Database class combining JSONStorage and Memory attributes.
    """
    
    def __init__(
        self,
        directory_path: str,
        pretty_print: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = JSONStorage(
            directory_path=directory_path,
            pretty_print=pretty_print
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)


class Mem0Database(DatabaseBase[Mem0Storage]):
    """
    Database class combining Mem0Storage and Memory attributes.
    """
    
    def __init__(
        self,
        client: Optional[Union['Mem0Memory', 'Mem0AsyncMemoryClient']] = None,
        api_key: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        local_config: Optional[Dict[str, Any]] = None,
        namespace: str = "upsonic",
        infer: bool = False,
        custom_categories: Optional[List[str]] = None,
        enable_caching: bool = True,
        cache_ttl: int = 300,
        output_format: str = "v1.1",
        version: str = "v2",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        storage = Mem0Storage(
            client=client,
            api_key=api_key,
            org_id=org_id,
            project_id=project_id,
            local_config=local_config,
            namespace=namespace,
            infer=infer,
            custom_categories=custom_categories,
            enable_caching=enable_caching,
            cache_ttl=cache_ttl,
            output_format=output_format,
            version=version
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            user_profile_schema=user_profile_schema,
            dynamic_user_profile=dynamic_user_profile,
            num_last_messages=num_last_messages,
            model=model,
            debug=debug,
            debug_level=debug_level,
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        super().__init__(storage=storage, memory=memory)
