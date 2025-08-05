from typing import Set

from upsonic.tasks.tasks import Task 
from upsonic.utils.package.exception import ModelCapabilityError
from upsonic.utils.file_helpers import get_clean_extension
from upsonic.models.model_registry import get_model_registry_entry

def validate_attachments_for_model(llm_model: str, single_task: Task) -> None:
    """
    Validates if the attachments in a task are supported by the specified model.

    This function is driven exclusively by the model's 'capabilities' entry
    in the MODEL_REGISTRY. It will raise an error if an attachment's file
    type is not explicitly listed in the model's supported extensions.

    Args:
        llm_model: The string identifier of the model (e.g., "gemini/gemini-1.5-pro").
        single_task: The Task object containing the list of attachments.

    Raises:
        ModelCapabilityError: If an attachment has an extension that is not
                              supported by the specified model.
        ValueError: If the model is not found in the registry.
    """
    if not single_task.attachments:
        return

    model_info = get_model_registry_entry(llm_model)
    if not model_info:
        raise ValueError(f"Model '{llm_model}' not found in MODEL_REGISTRY.")

    model_capabilities = model_info.get("capabilities", {})
    
    if not isinstance(model_capabilities, dict):
        supported_extensions_set: set[str] = set()
    else:
        supported_extensions_set: set[str] = {
            ext for ext_list in model_capabilities.values() for ext in ext_list
        }

    GENERIC_VIDEO_EXTS: Set[str] = {"mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp", "mkv"}
    GENERIC_IMAGE_EXTS: Set[str] = {"png", "jpeg", "jpg", "webp", "heic", "heif", "gif", "bmp"}
    GENERIC_AUDIO_EXTS: Set[str] = {"wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a"}

    for attachment_path in single_task.attachments:
        extension = get_clean_extension(attachment_path)

        if not extension:
            continue
        
        if extension in supported_extensions_set:
            continue

        required_capability = None
        if extension in GENERIC_VIDEO_EXTS:
            required_capability = "video"
        elif extension in GENERIC_IMAGE_EXTS:
            required_capability = "image"
        elif extension in GENERIC_AUDIO_EXTS:
            required_capability = "audio"


        if required_capability:
            supported_for_capability = model_capabilities.get(required_capability, []) if isinstance(model_capabilities, dict) else []
            raise ModelCapabilityError(
                model_name=llm_model,
                attachment_path=attachment_path,
                attachment_extension=extension,
                required_capability=required_capability,
                supported_extensions=supported_for_capability
            )
        else:
            raise ModelCapabilityError(
                model_name=llm_model,
                attachment_path=attachment_path,
                attachment_extension=extension,
                required_capability=None,
                supported_extensions=list(sorted(supported_extensions_set))
            )
        # If the file is not a recognized media type (e.g., .txt, .pdf), we ignore it.