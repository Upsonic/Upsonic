from __future__ import annotations
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field, model_validator

from upsonic.text_splitter.base import ChunkingStrategy



class RoutingRule(BaseModel):
    """
    A structured, validated rule for the RuleBasedChunkingStrategy.

    This model defines the conditions under which a specific chunking strategy
    should be applied to a document. It provides a robust and declarative way
    to configure document routing.
    """
    strategy: ChunkingStrategy = Field(
        ...,
        description="The ChunkingStrategy instance to use if this rule matches."
    )
    file_extension: Optional[str] = Field(
        default=None,
        description="A simple file extension to match against (e.g., '.md', '.py')."
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="A dictionary to perform exact matches against the document's metadata."
    )

    @model_validator(mode='after')
    def check_at_least_one_condition(self) -> 'RoutingRule':
        """Ensures that each rule has at least one condition to match against."""
        if self.file_extension is None and self.metadata_filter is None:
            raise ValueError("A RoutingRule must have at least one condition: either 'file_extension' or 'metadata_filter'.")
        return self
    
    class Config:
        arbitrary_types_allowed = True