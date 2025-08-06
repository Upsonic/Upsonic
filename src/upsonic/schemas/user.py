from pydantic import BaseModel, Field
from typing import List, Optional


class UserTraits(BaseModel):
    detected_expertise: Optional[str] = Field(
        None, 
        description="The user's expertise level on the topic, e.g., 'beginner', 'intermediate', 'expert'."
    )
    detected_tone: Optional[str] = Field(
        None,
        description="The user's preferred communication tone, e.g., 'formal', 'casual', 'technical'."
    )
    inferred_interests: Optional[List[str]] = Field(
        None,
        description="A list of topics or keywords the user seems interested in."
    )