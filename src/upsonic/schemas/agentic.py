from typing import List

from pydantic import BaseModel, Field



class PropositionList(BaseModel):
    """A Pydantic model to ensure the LLM returns a list of propositions."""
    propositions: List[str] = Field(..., description="A list of simple, self-contained factual statements extracted from the source text.")

class Topic(BaseModel):
    """Represents a single, emergent topic with its assigned propositions."""
    topic_id: int = Field(..., description="A unique integer ID for this topic.")
    propositions: List[str] = Field(..., description="A list of the propositions that belong to this topic.")

class TopicAssignmentList(BaseModel):
    """The structure for the LLM's response during the batch clustering stage."""
    topics: List[Topic] = Field(..., description="A list of all identified topics, each containing its assigned propositions.")

class RefinedTopic(BaseModel):
    """The final, refined metadata for a completed chunk."""
    title: str = Field(..., description="A concise, human-readable title for the topic cluster (e.g., 'Candidate Skills and Experience').")
    summary: str = Field(..., description="A one-sentence summary that encapsulates the core information of all propositions in the cluster.")