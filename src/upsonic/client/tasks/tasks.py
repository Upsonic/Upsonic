from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union


from .task_response import CustomTaskResponse, ObjectResponse
from ..printing import get_price_id_total_cost
class Task(BaseModel):
    description: str
    tools: list[Any] = []
    response_format: Union[Type[CustomTaskResponse], Type[ObjectResponse], None] = None
    _response: Any = None
    context: Any = None
    price_id_: str = None
    not_main_task: bool = False
    
    def __init__(self, description: str = None, **data):
        if description is not None:
            data["description"] = description
        super().__init__(**data)

    @property
    def price_id(self):
        if self.price_id_ is None:
            import uuid
            self.price_id_ = str(uuid.uuid4())
        return self.price_id_

    @property
    def response(self):

        if self._response is None:
            return None

        if type(self._response) == str:
            return self._response



        if self._response._upsonic_response_type == "custom":
            return self._response.output()
        else:
            return self._response



    def get_total_cost(self):
        if self.price_id_ is None:
            return None
        return get_price_id_total_cost(self.price_id)