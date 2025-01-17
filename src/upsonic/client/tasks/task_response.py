from typing import List
from pydantic import BaseModel


class ObjectResponse(BaseModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._upsonic_response_type = "default"

class CustomTaskResponse(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._upsonic_response_type = "custom"

    def output(self):
        # Get the first field name from the model
        field_name = next(iter(self.__annotations__))
        # Return the value of that field
        return getattr(self, field_name)




def IntResponse(name: str):

   
    name = name.lower().replace(" ", "_")

    
    class IntegerResponse(CustomTaskResponse):
        __annotations__ = {name: int}
        locals()[name] = None
    IntegerResponse.__name__ = name.capitalize()
    return IntegerResponse


def FloatResponse(name: str):

    
    name = name.lower().replace(" ", "_")
    class FloatingResponse(CustomTaskResponse):
        __annotations__ = {name: float}
        locals()[name] = None
    FloatingResponse.__name__ = name.capitalize()
    return FloatingResponse


def BoolResponse(name: str):

    
    name = name.lower().replace(" ", "_")   
    class BooleanResponse(CustomTaskResponse):
        __annotations__ = {name: bool}
        locals()[name] = None
    BooleanResponse.__name__ = name.capitalize()
    return BooleanResponse


def StrResponse(name: str):

    
    name = name.lower().replace(" ", "_")
    class StringResponse(CustomTaskResponse):
        __annotations__ = {name: str}
        locals()[name] = None
    StringResponse.__name__ = name.capitalize()
    return StringResponse

def StrInListResponse(name: str):
    name = name.lower().replace(" ", "_")
    class StrInListResponse(CustomTaskResponse):
        __annotations__ = {name: list[str]}
        locals()[name] = None
    StrInListResponse.__name__ = name.capitalize()
    return StrInListResponse