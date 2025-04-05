import copy
from datetime import datetime
import dill
import cloudpickle
cloudpickle.DEFAULT_PROTOCOL = 2
import base64

from pydantic import BaseModel
from ..knowledge_base.knowledge_base import KnowledgeBase
from ..printing import error_message
from ...exception import (
    NoAPIKeyException,
    ContextWindowTooSmallException,
    InvalidRequestException,
    UnsupportedLLMModelException,
    UnsupportedComputerUseModelException,
    CallErrorException
)


def serialize_context(context, client):
    if isinstance(context, KnowledgeBase):
        context = context.markdown(client)
    
    return context

def context_serializer(context, client):

    if context is None:
        context = []


    copy_of_context = copy.deepcopy(context)

    if not isinstance(copy_of_context, list):
        copy_of_context = [copy_of_context]
    

    # Adding current date time to the context
    copy_of_context.append(f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for i, each in enumerate(copy_of_context):
            try:
                each.tools = []
            except:
                pass
            try:
                each.response_format = None
            except:
                pass

            copy_of_context[i] = serialize_context(each, client)


    the_module = dill.detect.getmodule(copy_of_context)
    if the_module is not None:
        cloudpickle.register_pickle_by_value(the_module)


    pickled_context = cloudpickle.dumps(copy_of_context)
    context = base64.b64encode(pickled_context).decode("utf-8")


    return context



def response_format_serializer(response_format):
    if response_format is None:
        response_format_str = "str"
    elif isinstance(response_format, (type, BaseModel)):
        # If it's a Pydantic model or other type, cloudpickle and base64 encode it
        the_module = dill.detect.getmodule(response_format)
        if the_module is not None:
            cloudpickle.register_pickle_by_value(the_module)
        pickled_format = cloudpickle.dumps(response_format)
        response_format_str = base64.b64encode(pickled_format).decode("utf-8")
    else:
        response_format_str = "str"

    return response_format_str


def response_format_deserializer(response_format_str, result):
    if response_format_str != "str":
        decoded_result = base64.b64decode(result["result"])
        deserialized_result = cloudpickle.loads(decoded_result)
    else:
        deserialized_result = result["result"]

    result["result"] = deserialized_result

    return result


def tools_serializer(tools_):
    tools = []
    for i in tools_:


        if isinstance(i, type):

            tools.append(i.__name__+".*")
        # If its a string, get the name of the string
        elif isinstance(i, str):

            tools.append(i)
        elif isinstance(i, object):
            sub_i = i.__class__
            tools.append(sub_i.__name__+".*")
    return tools



def error_handler(result):
    if result["status_code"] == 401:
        error_message("API Key Error", result["detail"], 401)
        raise NoAPIKeyException(result["detail"])
    
    if result["status_code"] == 402:
        error_message("Context Window Error", result["detail"], 402)
        raise ContextWindowTooSmallException(result["detail"])

    if result["status_code"] == 403:
        error_message("Invalid Request", result["detail"], 403)
        raise InvalidRequestException(result["detail"])

    if result["status_code"] == 400:
        error_message("Unsupported Model", result["detail"], 400)
        raise UnsupportedLLMModelException(result["detail"])

    if result["status_code"] == 405:
        error_message("Unsupported Computer Use Model", result["detail"], 405)
        raise UnsupportedComputerUseModelException(result["detail"])

    if result["status_code"] == 500:
        # Extract meaningful message from the error if available
        error_detail = result.get("message", str(result))
        error_message("Call Error", error_detail, 500)
        return True  # Indicate this is a retriable error
        
    return False  # Not a retriable error