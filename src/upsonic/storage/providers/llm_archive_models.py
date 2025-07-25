import uuid
from sqlalchemy import (
    types,
    Column,
    String,
    Text,
    JSON,
    BigInteger,
    ForeignKey,
    Float,
    CHAR
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship


class GUID(types.TypeDecorator):
    """
    Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(32),
    storing as stringified hex values.
    """
    impl = types.CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(types.CHAR(32))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        elif dialect.name == 'postgresql':
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value).hex
            return value.hex

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        return value

Base = declarative_base()


class LLMConversationModel(Base):
    __tablename__ = "llm_conversations"

    conversation_id = Column(GUID, primary_key=True)
    user_id = Column(String(255), index=True)
    entity_id = Column(GUID, index=True)
    created_at = Column(BigInteger, nullable=False, index=True)
    updated_at = Column(BigInteger, nullable=False, index=True)
    summary = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    meta_data = Column("metadata", JSON, nullable=True)
    turns = relationship("LLMTurnModel", back_populates="conversation", cascade="all, delete-orphan")
    artifacts = relationship("LLMArtifactModel", back_populates="conversation", cascade="all, delete-orphan")


class LLMTurnModel(Base):
    __tablename__ = "llm_turns"

    turn_id = Column(GUID, primary_key=True)
    conversation_id = Column(GUID, ForeignKey("llm_conversations.conversation_id"), nullable=False, index=True)
    role = Column(String(50), nullable=False, index=True)
    timestamp_start = Column(Float, nullable=False, index=True)
    timestamp_end = Column(Float, nullable=True)
    turn_data = Column(JSON, nullable=False)
    conversation = relationship("LLMConversationModel", back_populates="turns")


class LLMArtifactModel(Base):
    __tablename__ = "llm_artifacts"

    artifact_id = Column(GUID, primary_key=True)
    conversation_id = Column(GUID, ForeignKey("llm_conversations.conversation_id"), nullable=False, index=True)
    turn_id = Column(GUID, nullable=True, index=True)
    mime_type = Column(String(255), nullable=False)
    storage_uri = Column(String(1024), nullable=False)
    meta_data = Column("metadata", JSON, nullable=True)
    conversation = relationship("LLMConversationModel", back_populates="artifacts")