from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorSearchResult:
    """
    A standardized data structure for a single search result.
    All provider implementations must return a list of these objects.
    """
    id: Union[str, int]
    score: float
    payload: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None
    text: Optional[str] = None