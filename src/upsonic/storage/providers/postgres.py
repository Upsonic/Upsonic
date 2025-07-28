import time
from typing import List, Optional
from pathlib import Path
import uuid
import os

from sqlalchemy import create_engine, inspect as sqlalchemy_inspect, text, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session as DBSession

from upsonic.storage.base import Storage, SchemaMismatchError
from upsonic.storage.settings import PostgresSettings
from upsonic.storage.session.llm import LLMConversation, LLMTurn, Artifact
from .llm_archive_models import Base, LLMConversationModel, LLMTurnModel, LLMArtifactModel


class PostgresStorage(Storage):
    """
    Production-grade storage provider for the LLM Interaction Archive using PostgreSQL.
    """
    def __init__(self, settings: PostgresSettings):
        super().__init__()
        self._set_mode(settings.STORAGE_MODE)
        self.artifacts_base_path = settings.POSTGRES_ARTIFACT_PATH.resolve()
        self.db_uri = str(settings.POSTGRES_DB_URI)
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[DBSession]] = None

    def _verify_schema(self):
        inspector = sqlalchemy_inspect(self._engine)
        required_tables = {"llm_conversations", "llm_turns", "llm_artifacts"}
        
        for table_name in required_tables:
            if not inspector.has_table(table_name):
                raise SchemaMismatchError(f"Missing required table: '{table_name}'.")

    def connect(self) -> None:
        if self._engine:
            return
        try:
            self._engine = create_engine(self.db_uri)
            Base.metadata.create_all(self._engine, checkfirst=True)
            self._verify_schema()
            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            with self._engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    def disconnect(self) -> None:
        if self._engine:
            self._engine.dispose()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Returns the current connection state."""
        return self._connected

    def _get_db_session(self) -> DBSession:
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
            turn_data_json = turn.model_dump(
                mode="json",
                exclude={"turn_id", "role", "timestamp_start", "timestamp_end"}
            )
            
            new_turn = LLMTurnModel(
                turn_id=uuid.UUID(turn.turn_id),
                conversation_id=uuid.UUID(conversation_id),
                role=turn.role,
                timestamp_start=turn.timestamp_start,
                timestamp_end=turn.timestamp_end,
                turn_data=turn_data_json
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
                turn_data = turn_model.turn_data
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
                metadata=convo_model.metadata,
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
                    metadata=r.metadata,
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
                metadata=artifact.metadata
            )
            db.add(new_artifact)
            db.commit()
        except Exception as e:
            db.rollback()
            raise IOError(f"Failed to log artifact {artifact.artifact_id}: {e}")
        finally:
            db.close()

    def drop(self) -> None:
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