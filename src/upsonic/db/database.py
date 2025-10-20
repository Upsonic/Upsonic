from __future__ import annotations
from typing import Optional, Type, Union, Dict, Any, List, Literal, Generic, TypeVar
from pydantic import BaseModel

from ..storage.base import Storage
from ..storage.memory.memory import Memory
from ..storage.providers import (
    InMemoryStorage,
    JSONStorage,
    Mem0Storage,
    PostgresStorage,
    RedisStorage,
    SqliteStorage,
    MongoStorage
)
from ..models import Model

# Generic type variable for storage providers
StorageType = TypeVar('StorageType', bound=Storage)


class DatabaseBase(Generic[StorageType]):
    """
    Base class for all database classes that combine storage providers with memory.
    
    This class provides a common interface and type safety for all database implementations.
    It uses generic types to ensure type safety while allowing different storage backends.
    """
    
    def __init__(
        self,
        storage: StorageType,
        memory: Memory
    ):
        """
        Initialize the database with storage and memory components.
        
        Args:
            storage: The storage provider instance
            memory: The memory instance
        """
        self.storage = storage
        self.memory = memory
    
    def __repr__(self) -> str:
        """String representation of the database instance."""
        return f"{self.__class__.__name__}(storage={type(self.storage).__name__}, memory={type(self.memory).__name__})"


class SqliteDatabase(DatabaseBase[SqliteStorage]):
    """
    Database class combining SqliteStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # SqliteStorage attributes
        sessions_table_name: str,
        profiles_table_name: str,
        db_file: Optional[str] = None,
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = SqliteStorage(
            sessions_table_name=sessions_table_name,
            profiles_table_name=profiles_table_name,
            db_file=db_file
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class PostgresDatabase(DatabaseBase[PostgresStorage]):
    """
    Database class combining PostgresStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # PostgresStorage attributes
        sessions_table_name: str,
        profiles_table_name: str,
        db_url: str,
        schema: str = "public",
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = PostgresStorage(
            sessions_table_name=sessions_table_name,
            profiles_table_name=profiles_table_name,
            db_url=db_url,
            schema=schema
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class MongoDatabase(DatabaseBase[MongoStorage]):
    """
    Database class combining MongoStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # MongoStorage attributes
        db_url: str,
        database_name: str,
        sessions_collection_name: str = "interaction_sessions",
        profiles_collection_name: str = "user_profiles",
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = MongoStorage(
            db_url=db_url,
            database_name=database_name,
            sessions_collection_name=sessions_collection_name,
            profiles_collection_name=profiles_collection_name
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class RedisDatabase(DatabaseBase[RedisStorage]):
    """
    Database class combining RedisStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # RedisStorage attributes
        prefix: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ssl: bool = False,
        expire: Optional[int] = None,
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = RedisStorage(
            prefix=prefix,
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            expire=expire
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class InMemoryDatabase(DatabaseBase[InMemoryStorage]):
    """
    Database class combining InMemoryStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # InMemoryStorage attributes
        max_sessions: Optional[int] = None,
        max_profiles: Optional[int] = None,
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = InMemoryStorage(
            max_sessions=max_sessions,
            max_profiles=max_profiles
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class JSONDatabase(DatabaseBase[JSONStorage]):
    """
    Database class combining JSONStorage and Memory attributes.
    """
    
    def __init__(
        self,
        # JSONStorage attributes
        directory_path: str,
        pretty_print: bool = True,
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = JSONStorage(
            directory_path=directory_path,
            pretty_print=pretty_print
        )
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)


class Mem0Database(DatabaseBase[Mem0Storage]):
    """
    Database class combining Mem0Storage and Memory attributes.
    """
    
    def __init__(
        self,
        # Mem0Storage attributes
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
        # Memory attributes
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
        feed_tool_call_results: bool = False,
        user_memory_mode: Literal['update', 'replace'] = 'update'
    ):
        # Initialize storage
        storage = Mem0Storage(
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
        
        # Initialize memory
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
            feed_tool_call_results=feed_tool_call_results,
            user_memory_mode=user_memory_mode
        )
        
        # Call parent constructor
        super().__init__(storage=storage, memory=memory)
