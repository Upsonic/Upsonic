from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BaseModelSettings(BaseModel):
    """
    A base model for universal, provider-agnostic model settings.

    This class defines the common parameters that are broadly supported across
    most large language models. Provider-specific settings classes should
    inherit from this to ensure a consistent configuration foundation.
    """

    model_name: Optional[str] = Field(
        default=None,
        description="The name of the model to use for this request."
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Controls randomness. Lower values make the model more deterministic.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to generate in the completion.",
        gt=0,
    )
    top_p: Optional[float] = Field(
        default=None,
        description=(
            "Nucleus sampling parameter. The model considers only the tokens "
            "comprising the top 'p' probability mass."
        ),
        ge=0.0,
        le=1.0,
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Penalty for new tokens based on whether they appear in the text so far, "
            "increasing the model's likelihood to talk about new topics."
        ),
        ge=-2.0,
        le=2.0,
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description=(
            "Penalty for new tokens based on their existing frequency in the text so far, "
            "decreasing the model's likelihood to repeat the same line verbatim."
        ),
        ge=-2.0,
        le=2.0,
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="A list of sequences where the API will stop generating further tokens."
    )
    timeout: Optional[float] = Field(
        default=None, description="The timeout for the API request in seconds.", gt=0
    )
    
    # A catch-all for any extra, non-defined parameters.
    # This allows for forward compatibility with new provider features.
    extra_body: Optional[Dict] = Field(
        default=None, description="Any additional parameters to be passed to the provider's API request body."
    )