"""User input field model and dynamic user control flow toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class UserInputField:
    """Represents a single field that requires user input.
    
    Used in both static user input (via @tool(requires_user_input=True))
    and dynamic user input (via UserControlFlowTools).
    """

    name: str
    field_type: str
    description: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "field_type": self.field_type,
            "description": self.description,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInputField":
        """Reconstruct from dictionary."""
        return cls(
            name=data["name"],
            field_type=data.get("field_type", "str"),
            description=data.get("description"),
            value=data.get("value"),
        )


def _build_user_input_schema_from_tool(
    tool_name: str,
    tool_args: Dict[str, Any],
    func: Any,
    user_input_fields: List[str],
) -> List[UserInputField]:
    """Build UserInputField list from a tool function's signature and config.
    
    Fields in user_input_fields get value=None (user must provide).
    Other fields get pre-filled from tool_args (agent-provided).
    If user_input_fields is empty, ALL fields require user input.
    """
    import inspect

    sig = inspect.signature(func) if func else None
    schema: List[UserInputField] = []

    if sig:
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                type_str = "str"
            else:
                type_str = getattr(annotation, "__name__", str(annotation))

            is_user_field = (not user_input_fields) or (param_name in user_input_fields)
            pre_filled_value = tool_args.get(param_name) if not is_user_field else None

            schema.append(UserInputField(
                name=param_name,
                field_type=type_str,
                description=None,
                value=pre_filled_value,
            ))
    else:
        for arg_name, arg_value in tool_args.items():
            is_user_field = (not user_input_fields) or (arg_name in user_input_fields)
            schema.append(UserInputField(
                name=arg_name,
                field_type=type(arg_value).__name__ if arg_value is not None else "str",
                description=None,
                value=None if is_user_field else arg_value,
            ))

    return schema


def _build_dynamic_user_input_schema(
    fields: List[Dict[str, str]],
) -> List[UserInputField]:
    """Build UserInputField list from dynamically-constructed field definitions.
    
    Used by the ``get_user_input`` tool in UserControlFlowTools.
    """
    schema: List[UserInputField] = []
    for f in fields:
        schema.append(UserInputField(
            name=f.get("field_name", f.get("name", "unknown")),
            field_type=f.get("field_type", "str"),
            description=f.get("field_description", f.get("description")),
            value=None,
        ))
    return schema


from upsonic.tools.base import ToolKit
from upsonic.tools.config import tool


class UserControlFlowTools(ToolKit):
    """Toolkit that gives agents the ability to dynamically request user input.

    When added to an agent's tools, this provides a ``get_user_input`` tool
    that the agent can call whenever it determines it needs information from
    the user.  The agent constructs the field list dynamically, and the
    framework pauses execution so the user can fill in the values.

    Instructions are injected into the agent's system prompt automatically
    via the ``ToolKit.instructions`` / ``ToolKit.add_instructions`` mechanism.

    Usage::

        from upsonic.tools.user_input import UserControlFlowTools

        agent = Agent("openai/gpt-4o", tools=[UserControlFlowTools()])
    """

    _DEFAULT_INSTRUCTIONS: str = (
        "You have access to the `get_user_input` tool to get user input for the given fields.\n"
        "\n"
        "1. **Get User Input**:\n"
        "    - Purpose: When you have call a tool/function where you don't have enough "
        "information, don't say you can't do it, just use the `get_user_input` tool to "
        "get the information you need from the user.\n"
        "    - Usage: Call `get_user_input` with the fields you require the user to fill "
        "in for you to continue your task.\n"
        "\n"
        "## IMPORTANT GUIDELINES\n"
        "- **Don't respond and ask the user for information.** Just use the `get_user_input` "
        "tool to get the information you need from the user.\n"
        "- **Don't make up information you don't have.** If you don't have the information, "
        "use the `get_user_input` tool to get the information you need from the user.\n"
        "- **Include only the required fields.** Include only the required fields in the "
        "`user_input_fields` parameter of the `get_user_input` tool. Don't include fields "
        "you already have the information for.\n"
        "- **Provide a clear and concise description of the field.** Clearly describe the "
        "field in the `field_description` parameter of the `user_input_fields` parameter "
        "of the `get_user_input` tool.\n"
        "- **Provide a type for the field.** Fill the `field_type` parameter of the "
        "`user_input_fields` parameter of the `get_user_input` tool with the type of the field.\n"
        "\n"
        "## INPUT VALIDATION AND CONVERSION\n"
        "- **Boolean fields**: Only explicit positive responses are considered True:\n"
        "  * True values: 'true', 'yes', 'y', '1', 'on', 't', 'True', 'YES', 'Y', 'T'\n"
        "  * False values: Everything else including 'false', 'no', 'n', '0', 'off', 'f', "
        "empty strings, unanswered fields, or any other input\n"
        "  * **CRITICAL**: Empty/unanswered fields should be treated as False (not selected)\n"
        "- **Users can leave fields unanswered.** Empty responses are valid and should be "
        "treated as False for boolean fields.\n"
        "- **NEVER ask for the same field twice.** Once you receive ANY user input for a "
        "field (including empty strings), accept it and move on.\n"
        "- **DO NOT validate or re-request input.** Accept whatever the user provides and "
        "convert it appropriately.\n"
        "- **Proceed with only the fields that were explicitly answered as True.** Skip or "
        "ignore fields that are False/unanswered.\n"
        "- **Complete the task immediately after receiving all user inputs, do not ask for "
        "confirmation or re-validation.**"
    )

    def __init__(
        self,
        instructions: Optional[str] = None,
        add_instructions: bool = True,
        enable_get_user_input: bool = True,
    ) -> None:
        _exclude: Optional[List[str]] = ["get_user_input"] if not enable_get_user_input else None
        super().__init__(
            exclude_tools=_exclude,
            instructions=instructions if instructions is not None else self._DEFAULT_INSTRUCTIONS,
            add_instructions=add_instructions,
        )

        self.enable_get_user_input: bool = enable_get_user_input

    @tool
    def get_user_input(self, fields: List[Dict[str, str]]) -> str:
        """Request input from the user for the specified fields.

        Each field should be a dict with:
        - field_name: identifier for the field
        - field_type: Python type (str, int, float, bool, list, dict)
        - field_description: description to help the user

        Args:
            fields: List of field definitions the agent needs from the user.
        """
        from upsonic.tools.hitl import UserInputPause

        schema = _build_dynamic_user_input_schema(fields)
        raise UserInputPause(user_input_schema=schema)
