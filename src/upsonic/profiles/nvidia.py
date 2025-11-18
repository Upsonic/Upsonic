from __future__ import annotations as _annotations

from upsonic.profiles import ModelProfile


def nvidia_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for NVIDIA's own models.
    
    This includes models like NVIDIA Nemotron and other NVIDIA-developed models.
    NVIDIA NIM also hosts models from other vendors which should use their respective profiles.
    """
    # NVIDIA's API support standard OpenAI-compatible features
    # through their NIM API
    return ModelProfile(
        supports_tools=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )

