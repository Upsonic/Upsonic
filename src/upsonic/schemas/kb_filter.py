"""Knowledge Base Filter Expression for Vector Database Queries."""

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field


class KBFilterExpr(BaseModel):
    """
    Filter expression for knowledge base vector database queries.
    
    This class encapsulates all vector search parameters that can be used
    to configure and filter knowledge base queries.
    
    Attributes:
        top_k: Maximum number of results to return
        alpha: Weight for hybrid search (0.0 = pure keyword, 1.0 = pure vector)
        fusion_method: Method for fusing vector and keyword search results
        similarity_threshold: Minimum similarity score for results (0.0 to 1.0)
        filter: Metadata filter for constraining search results
    """
    
    top_k: Optional[int] = Field(
        default=None,
        description="Maximum number of results to return from vector search"
    )
    
    alpha: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Weight for hybrid search (0.0 = pure keyword, 1.0 = pure vector)"
    )
    
    fusion_method: Optional[Literal['rrf', 'weighted']] = Field(
        default=None,
        description="Method for fusing vector and keyword search results"
    )
    
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results"
    )
    
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filter for constraining search results"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, excluding None values."""
        return {
            k: v for k, v in self.model_dump().items() 
            if v is not None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KBFilterExpr":
        """Deserialize from dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_task(cls, task: Any) -> Optional["KBFilterExpr"]:
        """
        Create KBFilterExpr from Task vector search parameters.
        
        Args:
            task: Task object with vector_search_* attributes
            
        Returns:
            KBFilterExpr if any parameters are set, None otherwise
        """
        params = {
            'top_k': getattr(task, 'vector_search_top_k', None),
            'alpha': getattr(task, 'vector_search_alpha', None),
            'fusion_method': getattr(task, 'vector_search_fusion_method', None),
            'similarity_threshold': getattr(task, 'vector_search_similarity_threshold', None),
            'filter': getattr(task, 'vector_search_filter', None),
        }
        
        if any(v is not None for v in params.values()):
            return cls(**params)
        return None

