from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.messages.messages import (
        BinaryContent,
        DocumentUrl,
        ImageUrl,
        ModelRequest,
    )


@dataclass
class AgentRunInput:
    """Input data for an agent run.
    
    Handles user prompts, images, and documents. URL types (ImageUrl, DocumentUrl)
    are automatically converted to BinaryContent for unified handling.
    """
    
    user_prompt: Union[str, List, Dict, "ModelRequest", "BaseModel", List["ModelRequest"]]
    images: Optional[List["BinaryContent"]] = None
    documents: Optional[List["BinaryContent"]] = None
    
    def __post_init__(self):
        """Convert any URL types to BinaryContent after initialization."""
        if self.images:
            self.images = [
                self._convert_url_to_binary(img) if self._is_url_type(img) else img
                for img in self.images
            ]
        if self.documents:
            self.documents = [
                self._convert_url_to_binary(doc) if self._is_url_type(doc) else doc
                for doc in self.documents
            ]
    
    @staticmethod
    def _is_url_type(content: Any) -> bool:
        """Check if content is a URL type that needs conversion."""
        from upsonic.messages.messages import DocumentUrl, ImageUrl
        return isinstance(content, (ImageUrl, DocumentUrl))
    
    @staticmethod
    def _convert_url_to_binary(url_content: Union["ImageUrl", "DocumentUrl"]) -> "BinaryContent":
        """Download URL content and convert to BinaryContent."""
        import httpx
        from upsonic.messages.messages import BinaryContent
        
        response = httpx.get(url_content.url)
        response.raise_for_status()
        return BinaryContent(
            data=response.content,
            media_type=url_content.media_type,
            identifier=url_content.identifier
        )
    
    @staticmethod
    async def _aconvert_url_to_binary(url_content: Union["ImageUrl", "DocumentUrl"]) -> "BinaryContent":
        """Async download URL content and convert to BinaryContent."""
        import httpx
        from upsonic.messages.messages import BinaryContent
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url_content.url)
            response.raise_for_status()
            return BinaryContent(
                data=response.content,
                media_type=url_content.media_type,
                identifier=url_content.identifier
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with raw values (no serialization)."""
        return {
            "user_prompt": self.user_prompt,
            "images": self.images,
            "documents": self.documents,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunInput":
        """Reconstruct from dictionary."""
        from upsonic.messages.messages import BinaryContent
        
        # Handle images (dict only)
        images = None
        images_data = data.get("images")
        if images_data:
            images = []
            for img in images_data:
                if isinstance(img, dict):
                    images.append(BinaryContent(
                        data=base64.b64decode(img["data"]) if isinstance(img["data"], str) else img["data"],
                        media_type=img.get("media_type", "application/octet-stream"),
                        identifier=img.get("identifier")
                    ))
        
        # Handle documents (dict only)
        documents = None
        documents_data = data.get("documents")
        if documents_data:
            documents = []
            for doc in documents_data:
                if isinstance(doc, dict):
                    documents.append(BinaryContent(
                        data=base64.b64decode(doc["data"]) if isinstance(doc["data"], str) else doc["data"],
                        media_type=doc.get("media_type", "application/octet-stream"),
                        identifier=doc.get("identifier")
                    ))
        
        return cls(
            user_prompt=data.get("user_prompt"),
            images=images,
            documents=documents
        )
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        import cloudpickle
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentRunInput":
        """Deserialize from bytes."""
        import cloudpickle
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)

