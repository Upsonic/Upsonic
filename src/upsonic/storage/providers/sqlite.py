import json
import os
import time
from typing import Any, List, Optional
from pathlib import Path
import uuid

from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.event import listen
from sqlalchemy.orm import sessionmaker, Session as DBSession

from upsonic.storage.base import Storage, SchemaMismatchError
from upsonic.storage.settings import SQLiteSettings
from upsonic.storage.session.llm import LLMConversation, LLMTurn, Artifact

from .llm_archive_models import Base, LLMConversationModel, LLMTurnModel, LLMArtifactModel


def _enable_foreign_keys(dbapi_con, connection_record):
    """Ensures a new SQLite connection has FK support enabled."""
    cursor = dbapi_con.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class SQLiteStorage(Storage):
    """
    Lightweight, serverless storage for the LLM Interaction Archive using SQLite.
    This version is self-sufficient, automatically creating and verifying the
    three-table relational schema on connection.
    """
    def __init__(self, settings: SQLiteSettings):
        """
        Initializes the SQLite storage provider from a settings object.
        """
        super().__init__()
        self._set_mode(settings.STORAGE_MODE)
        self.db_path = str(settings.SQLITE_DB_PATH)
        self.artifacts_base_path = settings.SQLITE_ARTIFACT_PATH.resolve()

        self.db_uri = f"sqlite:///{self.db_path}" if self.db_path != ":memory:" else "sqlite:///"
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[DBSession]] = None

    def _verify_schema(self):
        """
        Verifies that the existing database schema contains the required tables.
        """
        if not self._engine:
            raise ConnectionError("Cannot verify schema, storage is not connected.")
        
        inspector = sqlalchemy_inspect(self._engine)
        required_tables = {"llm_conversations", "llm_turns", "llm_artifacts"}
        
        existing_tables = inspector.get_table_names()
        
        missing_tables = required_tables - set(existing_tables)
        if missing_tables:
            raise SchemaMismatchError(f"Missing required tables: {missing_tables}. A manual migration or database reset is required.")

    def connect(self) -> None:
        """
        Establishes connection and handles schema creation/verification.
        """
        if self._engine:
            self._connected = True
            return
        try:
            if self.db_path != ":memory:":
                db_dir = os.path.dirname(os.path.abspath(self.db_path))
                os.makedirs(db_dir, exist_ok=True)

            self._engine = create_engine(self.db_uri)
            
            if self.db_path != ":memory:":
                listen(self._engine, "connect", _enable_foreign_keys)

            Base.metadata.create_all(self._engine, checkfirst=True)
            self._verify_schema()

            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQLite at {self.db_path}: {e}") from e

    def disconnect(self) -> None:
        """Closes all connections in the connection pool."""
        if self._engine:
            self._engine.dispose()
        self._connected = False

    def is_connected(self) -> bool:
        """Returns the current connection state."""
        return self._connected

    def _get_db_session(self) -> DBSession:
        """Provides a new database session."""
        if not self._session_factory:
            raise ConnectionError("Storage is not connected.")
        return self._session_factory()
        
    def start_conversation(self, conversation: LLMConversation) -> None:
        db = self._get_db_session()
        try:
            new_convo = LLMConversationModel(
                conversation_id=uuid.UUID(conversation.conversation_id),
                user_id=conversation.user_id,
                entity_id=uuid.UUID(conversation.entity_id) if conversation.entity_id else None,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
                summary=conversation.summary,
                tags=conversation.tags,
                meta_data=conversation.metadata
            )
            db.add(new_convo)
            db.commit()
        except Exception as e:
            db.rollback()
            raise IOError(f"Failed to start conversation {conversation.conversation_id}: {e}")
        finally:
            db.close()

    def append_turn(self, conversation_id: str, turn: LLMTurn) -> None:
        db = self._get_db_session()
        try:
            turn_data_json_str = turn.model_dump_json(
                exclude={"turn_id", "role", "timestamp_start", "timestamp_end"}
            )
            
            new_turn = LLMTurnModel(
                turn_id=uuid.UUID(turn.turn_id),
                conversation_id=uuid.UUID(conversation_id),
                role=turn.role,
                timestamp_start=turn.timestamp_start,
                timestamp_end=turn.timestamp_end,
                turn_data=turn_data_json_str
            )
            db.add(new_turn)
            
            db.query(LLMConversationModel).filter(
                LLMConversationModel.conversation_id == conversation_id
            ).update({"updated_at": int(time.time())})
            
            db.commit()
        except Exception as e:
            db.rollback()
            raise IOError(f"Failed to append turn to conversation {conversation_id}: {e}")
        finally:
            db.close()

    def get_conversation(self, conversation_id: str) -> Optional[LLMConversation]:
        db = self._get_db_session()
        try:
            convo_model = db.query(LLMConversationModel).filter(
                LLMConversationModel.conversation_id == conversation_id
            ).first()

            if not convo_model:
                return None
            
            turns_models = db.query(LLMTurnModel).filter(
                LLMTurnModel.conversation_id == conversation_id
            ).order_by(LLMTurnModel.timestamp_start).all()
            
            turns = []
            for turn_model in turns_models:
                turn_data = json.loads(turn_model.turn_data)
                turn_data.update({
                    "turn_id": str(turn_model.turn_id),
                    "role": turn_model.role,
                    "timestamp_start": turn_model.timestamp_start,
                    "timestamp_end": turn_model.timestamp_end
                })
                turns.append(LLMTurn.model_validate(turn_data))
            
            return LLMConversation(
                conversation_id=str(convo_model.conversation_id),
                user_id=convo_model.user_id,
                entity_id=convo_model.entity_id,
                created_at=convo_model.created_at,
                updated_at=convo_model.updated_at,
                summary=convo_model.summary,
                tags=convo_model.tags,
                metadata=convo_model.meta_data,
                turns=turns
            )
        finally:
            db.close()

    def list_conversations(self, user_id: Optional[str] = None, entity_id: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[LLMConversation]:
        db = self._get_db_session()
        try:
            query = db.query(LLMConversationModel).order_by(LLMConversationModel.updated_at.desc())
            if user_id:
                query = query.filter(LLMConversationModel.user_id == user_id)
            if entity_id:
                query = query.filter(LLMConversationModel.entity_id == entity_id)
            
            results = query.limit(limit).offset(offset).all()
            
            return [
                LLMConversation(
                    conversation_id=str(r.conversation_id),
                    user_id=r.user_id,
                    entity_id=r.entity_id,
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    summary=r.summary,
                    tags=r.tags,
                    metadata=r.meta_data,
                    turns=[]
                ) for r in results
            ]
        finally:
            db.close()
    
    def log_artifact(self, artifact: Artifact) -> None:
        db = self._get_db_session()
        try:
            new_artifact = LLMArtifactModel(
                artifact_id=uuid.UUID(artifact.artifact_id),
                conversation_id=uuid.UUID(artifact.conversation_id),
                turn_id=uuid.UUID(artifact.turn_id) if artifact.turn_id else None,
                mime_type=artifact.mime_type,
                storage_uri=artifact.storage_uri,
                meta_data=artifact.metadata
            )
            db.add(new_artifact)
            db.commit()
        except Exception as e:
            db.rollback()
            raise IOError(f"Failed to log artifact {artifact.artifact_id}: {e}")
        finally:
            db.close()

    def drop(self) -> None:
        """Drops all tables related to the LLM archive."""
        if self._engine:
            Base.metadata.drop_all(self._engine)

    def store_artifact_data(self, artifact_id: str, conversation_id: str, binary_data: bytes) -> str:
        """Stores the artifact as a file on the application server's disk."""
        try:
            self.artifacts_base_path.mkdir(parents=True, exist_ok=True)
            
            conv_artifact_path = self.artifacts_base_path / conversation_id
            conv_artifact_path.mkdir(exist_ok=True)
            
            file_path = conv_artifact_path / artifact_id
            
            with file_path.open("wb") as f:
                f.write(binary_data)
            
            return file_path.as_uri()
        except OSError as e:
            raise IOError(f"Failed to write artifact file to {file_path}: {e}")

    def retrieve_artifact_data(self, storage_uri: str) -> bytes:
        """Retrieves an artifact from a file URI stored in the database."""
        if not storage_uri.startswith("file://"):
            raise ValueError(f"This storage provider can only handle 'file://' URIs for artifacts.")
        
        file_path_str = storage_uri[len("file://"):]
        if os.name == 'nt' and file_path_str.startswith('/'):
            file_path_str = file_path_str[1:]
        
        file_path = Path(file_path_str)
        
        try:
            with file_path.open("rb") as f:
                return f.read()
        except FileNotFoundError:
            raise
        except OSError as e:
            raise IOError(f"Failed to read artifact file from {file_path}: {e}")