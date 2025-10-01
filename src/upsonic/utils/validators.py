from typing import Set, List, Dict, TYPE_CHECKING, Any, Union
from upsonic.models import Model
from upsonic.models import infer_model

from upsonic.utils.package.exception import ModelCapabilityError
from upsonic.utils.file_helpers import get_clean_extension

if TYPE_CHECKING:
    from upsonic.tasks.tasks import Task 

GENERIC_IMAGE_EXTS: Set[str] = {"png", "jpeg", "jpg", "webp", "heic", "heif", "gif", "bmp"}
GENERIC_AUDIO_EXTS: Set[str] = {"wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a"}
GENERIC_VIDEO_EXTS: Set[str] = {"mp4", "mpeg", "mpg", "mov", "avi", "flv", "webm", "wmv", "3gpp", "3gp", "mkv"}

# TODO: Implement proper implementation of validate_attachments_for_model
# Temporary stub for validate_attachments_for_model until proper implementation

def validate_attachments_for_model(model_provider: Union[Model, str], single_task: "Task") -> None:
    """
    Validates if the attachments in a task are supported by the model provider,
    checking both the general capability and the specific file extension.

    Args:
        model_provider: The instantiated model provider object (e.g., OpenAI, Gemini) or the model name.
        single_task: The Task object containing the list of attachments.

    Raises:
        ModelCapabilityError: If an attachment requires a capability or a specific
                              file extension not supported by the model.
    """
    if not single_task.attachments:
        return

    if isinstance(model_provider, str):
        model_provider = infer_model(model_provider)

    if not hasattr(model_provider, 'capabilities'):
        raise ModelCapabilityError(
            model_name=model_provider.model_name,
            attachment_path=single_task.attachments[0],
            attachment_extension=get_clean_extension(single_task.attachments[0]),
            required_capability="vision, audio, or video",
            supported_extensions=[]
        )

    supported_capabilities_dict = getattr(model_provider, 'capabilities', {})

    if not supported_capabilities_dict:
        raise ModelCapabilityError(
            model_name=model_provider.model_name,
            attachment_path=single_task.attachments[0],
            attachment_extension=get_clean_extension(single_task.attachments[0]),
            required_capability="vision, audio, or video",
            supported_extensions=[]
        )

    for attachment_path in single_task.attachments:
        extension = get_clean_extension(attachment_path)
        if not extension:
            continue

        required_capability = None
        if extension in GENERIC_IMAGE_EXTS: required_capability = "vision"
        elif extension in GENERIC_AUDIO_EXTS: required_capability = "audio"
        elif extension in GENERIC_VIDEO_EXTS: required_capability = "video"

        if not required_capability:
            continue

        if required_capability in supported_capabilities_dict:
            
            supported_extensions_for_cap = supported_capabilities_dict[required_capability]
            if extension in supported_extensions_for_cap:
                continue
            else:
                raise ModelCapabilityError(
                    model_name=model_provider.model_name,
                    attachment_path=attachment_path,
                    attachment_extension=extension,
                    required_capability=required_capability,
                    supported_extensions=sorted(supported_extensions_for_cap)
                )
        else:
            all_supported_extensions: List[str] = [
                ext for cap_exts in supported_capabilities_dict.values() for ext in cap_exts
            ]
            raise ModelCapabilityError(
                model_name=model_provider.model_name,
                attachment_path=attachment_path,
                attachment_extension=extension,
                required_capability=required_capability,
                supported_extensions=sorted(all_supported_extensions)
            )