import base64
from pydantic import BaseModel


from typing import Any, List, Dict, Optional, Type, Union


from .task_response import ObjectResponse
from ..printing import get_price_id_total_cost

from ..knowledge_base.knowledge_base import KnowledgeBase

class Task(BaseModel):
    description: str
    images: Optional[List[str]] = None
    tools: list[Any] = []
    response_format: Union[Type[ObjectResponse], Type[BaseModel], type[str], None] = str
    _response: Any = None
    context: Any = None
    price_id_: Optional[str] = None
    not_main_task: bool = False
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    agent: Optional[Any] = None
    response_lang: Optional[str] = None
    _tool_calls: List[Dict[str, Any]] = []



    def __init__(
        self, 
        description: str, 
        images: Optional[List[str]] = None,
        tools: list[Any] = None,
        response_format: Union[Type[ObjectResponse], Type[BaseModel], type[str], None] = str,
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
            "response_lang": response_lang,
            "_tool_calls": []
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



        return self._response



    def get_total_cost(self):
        if self.price_id_ is None:
            return None
        return get_price_id_total_cost(self.price_id)
    
    @property
    def total_cost(self) -> Optional[float]:
        """
        Get the total estimated cost of this task.
        
        Returns:
            Optional[float]: The estimated cost in USD, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "estimated_cost" in the_total_cost:
            return the_total_cost["estimated_cost"]
        return None
        
    @property
    def total_input_token(self) -> Optional[int]:
        """
        Get the total number of input tokens used by this task.
        
        Returns:
            Optional[int]: The number of input tokens, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "input_tokens" in the_total_cost:
            return the_total_cost["input_tokens"]
        return None
        
    @property
    def total_output_token(self) -> Optional[int]:
        """
        Get the total number of output tokens used by this task.
        
        Returns:
            Optional[int]: The number of output tokens, or None if not available
        """
        the_total_cost = self.get_total_cost()
        if the_total_cost and "output_tokens" in the_total_cost:
            return the_total_cost["output_tokens"]
        return None

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get all tool calls made during this task's execution.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about tool calls,
            including tool name, parameters, and result.
        """
        return self._tool_calls
        
    def add_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """
        Add a tool call to the task's history.
        
        Args:
            tool_call (Dict[str, Any]): Dictionary containing information about the tool call.
                Should include 'tool_name', 'params', and 'tool_result' keys.
        """
        self._tool_calls.append(tool_call)
