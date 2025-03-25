import base64
from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union


from .task_response import CustomTaskResponse, ObjectResponse
from ..printing import get_price_id_total_cost

from ..knowledge_base.knowledge_base import KnowledgeBase

class Task(BaseModel):
    description: str
    images: Optional[List[str]] = None
    tools: list[Any] = []
    response_format: Union[Type[CustomTaskResponse], Type[ObjectResponse], None] = None
    _response: Any = None
    context: Any = None
    price_id_: Optional[str] = None
    not_main_task: bool = False
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    agent: Optional[Any] = None
    response_lang: Optional[str] = None



    def __init__(
        self, 
        description: str, 
        images: Optional[List[str]] = None,
        tools: list[Any] = None,
        response_format: Union[Type[CustomTaskResponse], Type[ObjectResponse], None] = None,
        response: Any = None,
        context: Any = None,
        price_id_: Optional[str] = None,
        not_main_task: bool = False,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        agent: Optional[Any] = None,
        response_lang: Optional[str] = None,
        **data
    ):
        if description is not None:
            data["description"] = description
            
        if tools is None:
            tools = []
            
        data.update({
            "images": images,
            "tools": tools,
            "response_format": response_format,
            "_response": response,
            "context": context,
            "price_id_": price_id_,
            "not_main_task": not_main_task,
            "start_time": start_time,
            "end_time": end_time,
            "agent": agent,
            "response_lang": response_lang
        })
        
        super().__init__(**data)
        self.validate_tools()

    @property
    def duration(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def validate_tools(self):
        """
        Validates each tool in the tools list.
        If a tool is a class and has a __control__ method, runs that method to verify it returns True.
        Raises an exception if the __control__ method returns False or raises an exception.
        """
        if not self.tools:
            return
            
        for tool in self.tools:
            # Check if the tool is a class
            if isinstance(tool, type) or hasattr(tool, '__class__'):
                # Check if the class has a __control__ method
                if hasattr(tool, '__control__') and callable(getattr(tool, '__control__')):

                        control_result = tool.__control__()


    
    async def additional_description(self, client):
        if not self.context:
            return ""
        
            
        rag_results = []
        for context in self.context:
            
            if isinstance(context, KnowledgeBase) and context.rag == True:
                await context.setup_rag(client)
                rag_results.append(await context.query(self.description))
                
        if rag_results:
            return f"The following is the RAG data: <rag>{' '.join(rag_results)}</rag>"
        return ""


    @property
    def images_base_64(self):
        if self.images is None:
            return None
        base_64_images = []
        for image in self.images:
            with open(image, "rb") as image_file:
                base_64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
        return base_64_images

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
    
    @property
    def total_cost(self) -> Optional[float]:
        total_task_cost = None
        the_total_cost = self.get_total_cost()
        if the_total_cost:
            if "estimated_cost" in the_total_cost:
                total_task_cost = the_total_cost["estimated_cost"]


        return total_task_cost