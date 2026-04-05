from __future__ import annotations
from typing import Optional, Type, Union, Dict, Any, List, Literal, Generic, TypeVar, TYPE_CHECKING
from pydantic import BaseModel

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
    from pymongo import MongoClient
    from redis import Redis
    from mem0 import Memory as Mem0Memory, MemoryClient as Mem0MemoryClient
    from mem0.configs.base import MemoryConfig
    from ..storage.mem0 import Mem0Storage
    from ..storage.postgres import PostgresStorage
    from ..storage.redis import RedisStorage
    from ..storage.sqlite import SqliteStorage
    from ..storage.mongo import MongoStorage
    from ..storage.in_memory import InMemoryStorage
    from ..storage.json import JSONStorage

from ..storage.base import Storage
from ..storage.memory.memory import Memory
from ..models import Model
from ..storage import InMemoryStorage, JSONStorage


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


class SqliteDatabase(DatabaseBase["SqliteStorage"]):
    """
    Database class combining SqliteStorage and Memory attributes.
    """

    def __init__(
        self,
        db_file: Optional[str] = None,
        db_engine: Optional[Any] = None,
        db_url: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        try:
            from ..storage import SqliteStorage
        except AttributeError as e:
            raise ImportError(
                "SqliteStorage is not available. Install optional dependency: uv sync --extra sqlite-storage"
            ) from e
        storage = SqliteStorage(
            db_file=db_file,
            db_engine=db_engine,
            db_url=db_url,
            session_table=session_table,
            user_memory_table=user_memory_table
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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


class PostgresDatabase(DatabaseBase["PostgresStorage"]):
    """
    Database class combining PostgresStorage and Memory attributes.
    """

    def __init__(
        self,
        db_url: Optional[str] = None,
        db_engine: Optional[Any] = None,
        db_schema: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        create_schema: bool = True,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        try:
            from ..storage import PostgresStorage
        except AttributeError as e:
            raise ImportError(
                "PostgresStorage is not available. Install optional dependency: uv sync --extra postgres-storage"
            ) from e
        storage = PostgresStorage(
            db_url=db_url,
            db_engine=db_engine,
            db_schema=db_schema,
            session_table=session_table,
            user_memory_table=user_memory_table,
            create_schema=create_schema
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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


class MongoDatabase(DatabaseBase["MongoStorage"]):
    """
    Database class combining MongoStorage and Memory attributes.
    """

    def __init__(
        self,
        db_client: Optional[Any] = None,
        db_name: Optional[str] = None,
        db_url: Optional[str] = None,
        session_collection: Optional[str] = None,
        user_memory_collection: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        try:
            from ..storage import MongoStorage
        except AttributeError as e:
            raise ImportError(
                "MongoStorage is not available. Install optional dependency: uv sync --extra mongo-storage"
            ) from e
        storage = MongoStorage(
            db_client=db_client,
            db_name=db_name,
            db_url=db_url,
            session_collection=session_collection,
            user_memory_collection=user_memory_collection
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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


class RedisDatabase(DatabaseBase["RedisStorage"]):
    """
    Database class combining RedisStorage and Memory attributes.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        db_url: Optional[str] = None,
        db_prefix: str = "upsonic",
        expire: Optional[int] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        try:
            from ..storage import RedisStorage
        except AttributeError as e:
            raise ImportError(
                "RedisStorage is not available. Install optional dependency: uv sync --extra redis-storage"
            ) from e
        storage = RedisStorage(
            redis_client=redis_client,
            db_url=db_url,
            db_prefix=db_prefix,
            expire=expire,
            session_table=session_table,
            user_memory_table=user_memory_table
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
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
            session_table=session_table,
            user_memory_table=user_memory_table
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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
        db_path: Optional[str] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
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
            db_path=db_path,
            session_table=session_table,
            user_memory_table=user_memory_table
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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


class Mem0Database(DatabaseBase["Mem0Storage"]):
    """
    Database class combining Mem0Storage and Memory attributes.
    """

    def __init__(
        self,
        memory_client: Optional[Any] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
        config: Optional[Any] = None,
        session_table: Optional[str] = None,
        user_memory_table: Optional[str] = None,
        default_user_id: str = "upsonic_default",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_session_memory: bool = False,
        summary_memory: bool = False,
        user_analysis_memory: bool = False,
        load_full_session_memory: Optional[bool] = None,
        load_summary_memory: Optional[bool] = None,
        load_user_analysis_memory: Optional[bool] = None,
        user_profile_schema: Optional[Type[BaseModel]] = None,
        dynamic_user_profile: bool = False,
        num_last_messages: Optional[int] = None,
        model: Optional[Union[Model, str]] = None,
        debug: bool = False,
        debug_level: int = 1,
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        try:
            from ..storage import Mem0Storage
        except AttributeError as e:
            raise ImportError(
                "Mem0Storage is not available. Install optional dependency: uv sync --extra mem0-storage"
            ) from e
        storage = Mem0Storage(
            memory_client=memory_client,
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            config=config,
            session_table=session_table,
            user_memory_table=user_memory_table,
            default_user_id=default_user_id
        )
        
        memory = Memory(
            storage=storage,
            session_id=session_id,
            user_id=user_id,
            full_session_memory=full_session_memory,
            summary_memory=summary_memory,
            user_analysis_memory=user_analysis_memory,
            load_full_session_memory=load_full_session_memory,
            load_summary_memory=load_summary_memory,
            load_user_analysis_memory=load_user_analysis_memory,
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
