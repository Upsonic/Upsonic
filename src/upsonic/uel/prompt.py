from __future__ import annotations

import re
from typing import Any, Optional, List, Tuple, Union, Literal, Dict

from upsonic.uel.runnable import Runnable
from upsonic.messages import ModelRequest, SystemPromptPart, UserPromptPart, ModelResponse, ModelMessage


class ChatPromptTemplate(Runnable[dict[str, Any], Union[str, ModelRequest]]):
    """A prompt template that formats variables into a prompt string or ModelRequest.
    
    This component takes a template string with placeholders (in {variable} format)
    and formats it with the provided input variables. It can also handle message-based
    templates with system/human/ai roles.
    
    Example:
        ```python
        # String template
        template = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        prompt = template.invoke({"topic": "bears"})
        # Returns: "Tell me a joke about bears"
        
        # Message template
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "Tell me about {topic}")
        ])
        request = template.invoke({"topic": "AI"})
        # Returns: ModelRequest with SystemPromptPart and UserPromptPart
        ```
    """
    
    def __init__(
        self, 
        template: Optional[str] = None, 
        input_variables: Optional[list[str]] = None,
        messages: Optional[List[Union[Tuple[str, str], Tuple[Literal["placeholder"], Dict[str, Any]]]]] = None
    ):
        """Initialize the prompt template.
        
        Args:
            template: The template string with {variable} placeholders
            input_variables: List of variable names in the template
            messages: List of (role, template) tuples for message-based templates
        """
        self.template = template
        self.input_variables = input_variables or []
        self.messages = messages
        self.is_message_template = messages is not None
        
        # Extract variables from messages if needed
        if self.is_message_template and not self.input_variables:
            all_vars = set()
            for msg in self.messages:
                if isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    if role == "placeholder" and isinstance(content, dict):
                        # Add placeholder variable
                        var_name = content.get("variable_name")
                        if var_name:
                            all_vars.add(var_name)
                    elif isinstance(content, str):
                        # Extract variables from template string
                        all_vars.update(re.findall(r'\{(\w+)\}', content))
            self.input_variables = list(all_vars)
    
    @classmethod
    def from_template(cls, template: str) -> "ChatPromptTemplate":
        """Create a ChatPromptTemplate from a template string.
        
        Automatically extracts variable names from {variable} placeholders.
        Creates a message-based template with a single human message.
        
        Args:
            template: The template string with {variable} placeholders
            
        Returns:
            A new ChatPromptTemplate instance
            
        Example:
            ```python
            template = ChatPromptTemplate.from_template(
                "Tell me a {adjective} joke about {topic}"
            )
            result = template.invoke({"adjective": "funny", "topic": "bears"})
            # Returns: ModelRequest with UserPromptPart(content="Tell me a funny joke about bears")
            ```
        """
        # Extract variable names from {variable} placeholders
        input_variables = re.findall(r'\{(\w+)\}', template)
        
        # Create a message-based template with single human message
        messages = [("human", template)]
        
        return cls(
            template=template,
            input_variables=input_variables,
            messages=messages
        )
    
    @classmethod
    def from_messages(cls, messages: List[Union[Tuple[str, str], Tuple[Literal["placeholder"], Dict[str, Any]]]]) -> "ChatPromptTemplate":
        """Create a ChatPromptTemplate from a list of message tuples.
        
        Supports:
        - ("system", "template"): System message
        - ("human", "template"): Human/user message
        - ("ai", "template"): AI/assistant message (for few-shot examples)
        - ("placeholder", {"variable_name": "chat_history"}): Placeholder for dynamic content
        
        Args:
            messages: List of (role, template) tuples
            
        Returns:
            A new ChatPromptTemplate instance
            
        Example:
            ```python
            template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant"),
                ("placeholder", {"variable_name": "chat_history"}),
                ("human", "Tell me about {topic}")
            ])
            ```
        """
        # Store all messages including placeholders
        return cls(messages=messages)
    
    def invoke(self, input: Union[dict[str, Any], ModelMessage, List[ModelMessage]], config: Optional[dict[str, Any]] = None) -> Union[str, ModelMessage, List[ModelMessage]]:
        """Format the template with the provided variables.
        
        Args:
            input: Dictionary of variable names to values, or a ModelMessage/list of ModelMessages to pass through
            config: Optional runtime configuration (unused)
            
        Returns:
            If input is a ModelMessage or list of ModelMessages: Returns the input unchanged
            For string templates: The formatted prompt string
            For message templates: A ModelRequest or list of ModelRequest/ModelResponse objects
            
        Raises:
            KeyError: If a required variable is missing from input
            TypeError: If input is not a dictionary, ModelMessage, or list of ModelMessages
        """
        # If input is a ModelMessage, return it directly without modification
        # Check by checking if it has the 'kind' attribute that ModelMessages have
        if hasattr(input, 'kind') and input.kind in ('request', 'response'):
            return input
        
        # If input is a list of ModelMessages, return it directly without modification
        if isinstance(input, list) and len(input) > 0:
            # Check if all items are ModelMessages by checking for 'kind' attribute
            if all(hasattr(item, 'kind') and item.kind in ('request', 'response') for item in input):
                return input
        
        # Otherwise, expect a dictionary
        if not isinstance(input, dict):
            raise TypeError(f"ChatPromptTemplate expects a dict, ModelMessage, or list of ModelMessages, got {type(input)}")
        
        # Check for missing variables
        missing_vars = set(self.input_variables) - set(input.keys())
        if missing_vars:
            raise KeyError(f"Missing required variables: {missing_vars}")
        
        # Handle message-based templates
        if self.is_message_template:
            from upsonic.messages import TextPart
            from upsonic.usage import RequestUsage
            from upsonic._utils import now_utc
            import datetime
            
            # Build the conversation history following strict rules:
            # 1. Only ONE SystemPromptPart in the FIRST ModelRequest
            # 2. One UserPromptPart per ModelRequest
            # 3. One TextPart per ModelResponse
            # 4. Request → Response → Request → Response sequence
            
            conversation_messages = []
            system_prompt = None
            current_request_parts = []
            has_system_in_first = False
            
            for msg in self.messages:
                if isinstance(msg, tuple) and len(msg) == 2:
                    role, content = msg
                    
                    # Handle placeholder injection
                    if role == "placeholder" and isinstance(content, dict):
                        var_name = content.get("variable_name")
                        if var_name and var_name in input:
                            # Inject chat history or other dynamic content
                            placeholder_value = input[var_name]
                            if isinstance(placeholder_value, list):
                                # Handle chat history as list of tuples (role, content)
                                for hist_item in placeholder_value:
                                    if isinstance(hist_item, tuple) and len(hist_item) == 2:
                                        hist_role, hist_content = hist_item
                                        
                                        # Flush current request if we have parts
                                        if current_request_parts:
                                            # Add system prompt to first request if we have one
                                            if system_prompt and not has_system_in_first:
                                                current_request_parts.insert(0, system_prompt)
                                                has_system_in_first = True
                                            
                                            conversation_messages.append(
                                                ModelRequest(parts=current_request_parts)
                                            )
                                            current_request_parts = []
                                        
                                        if hist_role == "human" or hist_role == "user":
                                            # Create a ModelRequest for human message
                                            request_parts = []
                                            # Add system prompt to first request only
                                            if system_prompt and not has_system_in_first:
                                                request_parts.append(system_prompt)
                                                has_system_in_first = True
                                            
                                            request_parts.append(
                                                UserPromptPart(
                                                    content=hist_content,
                                                    timestamp=now_utc()
                                                )
                                            )
                                            conversation_messages.append(
                                                ModelRequest(parts=request_parts)
                                            )
                                        elif hist_role == "ai" or hist_role == "assistant":
                                            # Create a ModelResponse for AI message
                                            conversation_messages.append(
                                                ModelResponse(
                                                    parts=[TextPart(
                                                        content=hist_content
                                                    )],
                                                    usage=RequestUsage(input_tokens=0, output_tokens=0),
                                                    model_name="unknown",  # We don't know the model from history
                                                    timestamp=now_utc()
                                                )
                                            )
                                        elif hist_role == "system":
                                            # Store system prompt for first request
                                            if not system_prompt:
                                                system_prompt = SystemPromptPart(
                                                    content=hist_content,
                                                    timestamp=now_utc()
                                                )
                            elif isinstance(placeholder_value, str):
                                # If it's a string, add as user prompt
                                # Check if we already have a UserPromptPart and combine them
                                if current_request_parts and isinstance(current_request_parts[-1], UserPromptPart):
                                    # Combine with existing UserPromptPart
                                    current_request_parts[-1] = UserPromptPart(
                                        content=current_request_parts[-1].content + " " + placeholder_value,
                                        timestamp=now_utc()
                                    )
                                else:
                                    # Create new UserPromptPart
                                    current_request_parts.append(
                                        UserPromptPart(
                                            content=placeholder_value,
                                            timestamp=now_utc()
                                        )
                                    )
                        continue
                    
                    # Format regular message templates
                    if isinstance(content, str):
                        try:
                            formatted_content = content.format(**input)
                        except KeyError as e:
                            raise KeyError(f"Template variable {e} not found in input: {input}")
                        
                        # Convert role to appropriate message part with timestamp
                        if role == "system":
                            # Store system prompt for first request
                            if not system_prompt:
                                system_prompt = SystemPromptPart(
                                    content=formatted_content,
                                    timestamp=now_utc()
                                )
                        elif role == "human" or role == "user":
                            # Check if we already have a UserPromptPart and combine them
                            if current_request_parts and isinstance(current_request_parts[-1], UserPromptPart):
                                # Combine with existing UserPromptPart
                                current_request_parts[-1] = UserPromptPart(
                                    content=current_request_parts[-1].content + " " + formatted_content,
                                    timestamp=now_utc()
                                )
                            else:
                                # Create new UserPromptPart
                                current_request_parts.append(
                                    UserPromptPart(
                                        content=formatted_content,
                                        timestamp=now_utc()
                                    )
                                )
                        elif role == "ai" or role == "assistant":
                            # For AI messages in few-shot examples, create ModelResponse
                            # First flush any accumulated parts
                            if current_request_parts:
                                # Add system prompt to first request if we have one
                                if system_prompt and not has_system_in_first:
                                    current_request_parts.insert(0, system_prompt)
                                    has_system_in_first = True
                                
                                conversation_messages.append(
                                    ModelRequest(parts=current_request_parts)
                                )
                                current_request_parts = []
                            
                            # Add the AI response
                            conversation_messages.append(
                                ModelResponse(
                                    parts=[TextPart(
                                        content=formatted_content
                                    )],
                                    usage=RequestUsage(input_tokens=0, output_tokens=0),
                                    model_name="unknown",
                                    timestamp=now_utc()
                                )
                            )
                        else:
                            raise ValueError(f"Unknown message role: {role}")
            
            # Add any remaining parts as a final ModelRequest
            if current_request_parts:
                # Add system prompt to first request if we have one
                if system_prompt and not has_system_in_first:
                    current_request_parts.insert(0, system_prompt)
                    has_system_in_first = True
                
                conversation_messages.append(
                    ModelRequest(parts=current_request_parts)
                )
            
            # Handle case where we only have a system prompt and no other parts
            if not conversation_messages and system_prompt:
                conversation_messages.append(
                    ModelRequest(parts=[system_prompt])
                )
            
            # Return the conversation history
            if len(conversation_messages) == 1:
                # Single message, return it directly
                return conversation_messages[0]
            else:
                # Multiple messages, return the list
                return conversation_messages
        
        # Handle string templates
        try:
            formatted = self.template.format(**input)
        except KeyError as e:
            raise KeyError(f"Template variable {e} not found in input: {input}")
        
        return formatted
    
    async def ainvoke(self, input: Union[dict[str, Any], ModelMessage, List[ModelMessage]], config: Optional[dict[str, Any]] = None) -> Union[str, ModelMessage, List[ModelMessage]]:
        """Format the template asynchronously.
        
        Since formatting is synchronous, this just calls invoke().
        
        Args:
            input: Dictionary of variable names to values, or a ModelMessage/list of ModelMessages to pass through
            config: Optional runtime configuration
            
        Returns:
            If input is a ModelMessage or list of ModelMessages: Returns the input unchanged
            For string templates: The formatted prompt string
            For message templates: A ModelRequest or list of ModelRequest/ModelResponse objects
        """
        return self.invoke(input, config)
    
    def __repr__(self) -> str:
        """Return a string representation of the template."""
        if self.is_message_template:
            return f"ChatPromptTemplate(messages={len(self.messages) if self.messages else 0} items)"
        return f"ChatPromptTemplate(template='{self.template}')"

  
