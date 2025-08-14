from __future__ import annotations

import functools
import hashlib
import inspect
import json
import re
import time
from pathlib import Path
from typing import Callable, Any, Generator, Tuple, Dict, TYPE_CHECKING, List, Optional
import asyncio
import copy

from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

from .tool import ToolConfig, ToolKit
from upsonic.tasks.tasks import Task
from upsonic.utils.printing import console, spacing, print_orchestrator_tool_step
from upsonic.tools.pseudo_tools import plan_and_execute
from upsonic.tools.thought import Thought, AnalysisResult


if TYPE_CHECKING:
    from upsonic.agent.agent import Direct


class ToolValidationError(Exception):
    """Custom exception raised for invalid tool definitions."""
    pass


class ToolProcessor:
    """
    The internal engine for inspecting, validating, normalizing, and wrapping
    user-provided tools into a format the agent can execute.
    """
    def __init__(self, agent: Optional['Direct'] = None):
        """
        Initializes the ToolProcessor.
        Args:
            agent: An optional instance of the Direct agent to enable context-aware
                   features like tool call limits.
        """
        self.agent_tool = agent

    def _validate_function(self, func: Callable):
        """
        Inspects a function to ensure it meets the requirements for a valid tool.
        A valid tool must have type hints for all parameters, a return type hint,
        and a non-empty docstring.
        Raises:
            ToolValidationError: If the function fails validation.
        """
        signature = inspect.signature(func)

        for param_name, param in signature.parameters.items():
            if param.name in ('self', 'cls'):
                continue
            if param.annotation is inspect.Parameter.empty:
                raise ToolValidationError(
                    f"Tool '{func.__name__}' is missing a type hint for parameter '{param_name}'."
                )

        if signature.return_annotation is inspect.Signature.empty:
            raise ToolValidationError(
                f"Tool '{func.__name__}' is missing a return type hint."
            )

        if not inspect.getdoc(func):
            raise ToolValidationError(
                f"Tool '{func.__name__}' is missing a docstring. The docstring is required to explain the tool's purpose to the LLM."
            )

    def generate_orchestrator_wrapper(self, original_agent_instance: 'Direct', task: 'Task') -> Callable:
        """
        Generates a highly specialized wrapper for the 'plan_and_execute' tool.

        This wrapper acts as the main orchestration engine. It takes a high-level plan
        of tool calls from the LLM. If reasoning is enabled, it automatically injects
        a mandatory analysis step after each tool call, creating an 'Act-then-Analyze' loop.
        """
        async def orchestrator_wrapper(thought: Thought) -> Any:
            console.print("[bold magenta]Orchestrator Activated:[/bold magenta] Received initial plan.")
            spacing()
            
            agent = self.agent_tool
            if not agent:
                return "Error: Orchestrator wrapper was not properly initialized with an agent instance."

            is_reasoning_enabled = agent.enable_reasoning_tool

            original_user_request = task.description
            execution_history = f"Orchestrator's execution history for the user's request:\n"
            execution_history += f"Initial Thought & Plan: {thought.plan}\n and Reasoning: {thought.reasoning}\n\n"
            pending_plan = thought.plan
            program_counter = 0

            all_tools = {
                tool_func.__name__: tool_func 
                for tool_func in original_agent_instance._upsonic_wrapped_tools.values()
                if tool_func.__name__ != 'plan_and_execute'
            }

            while program_counter < len(pending_plan):
                step = pending_plan[program_counter]
                
                tool_name = step.tool_name
                params = step.parameters
                tool_name = tool_name.split('.')[-1]

                console.print(f"[bold blue]Executing Tool Step {program_counter + 1}/{len(pending_plan)}:[/bold blue] Calling tool [cyan]{tool_name}[/cyan] with params {params}")
                
                if tool_name not in all_tools:
                    result = f"Error: Tool '{tool_name}' is not an available tool."
                    console.print(f"[bold red]{result}[/bold red]")
                else:
                    try:
                        tool_to_call = all_tools[tool_name]
                        result = await tool_to_call(**params)
                    except Exception as e:
                        error_message = f"An error occurred while executing tool '{tool_name}': {e}"
                        console.print(f"[bold red]{error_message}[/bold red]")
                        result = error_message

                print_orchestrator_tool_step(tool_name, params, result)
                execution_history += f"\nStep {program_counter + 1} (Tool: {tool_name}):\nResult: {result}\n"

                if is_reasoning_enabled:
                    console.print(f"[bold yellow]Injecting Mandatory Analysis Step after Tool '{tool_name}'...[/bold yellow]")
                    
                    analysis_prompt = (
                        f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{original_user_request}'\n\n"
                        "You are in the middle of a multi-step plan. An action has just been completed. You must now analyze the outcome before proceeding. "
                        "Based on the execution history, evaluate the result of the last tool call and decide the "
                        "most logical next action.\n\n"
                        "<ExecutionHistory>\n"
                        f"{execution_history}"
                        "</ExecutionHistory>"
                    )

                    analysis_task = Task(description=analysis_prompt, not_main_task=True, response_format=AnalysisResult)
                    analysis_agent = copy.copy(agent)
                    analysis_agent.enable_thinking_tool = False
                    analysis_agent.enable_reasoning_tool = False

                    analysis_result: AnalysisResult = await analysis_agent.do_async(analysis_task)
                    execution_history += f"\n--- Injected Analysis ---\nEvaluation: {analysis_result.evaluation}\n"

                    if analysis_result.next_action == 'continue_plan':
                        console.print("[bold green]Analysis complete. Continuing with the original plan.[/bold green]")
                        program_counter += 1
                        continue
                    
                    elif analysis_result.next_action == 'final_answer':
                        console.print("[bold green]Analysis concluded that the task is complete. Proceeding to final synthesis.[/bold green]")
                        break
                    
                    elif analysis_result.next_action == 'revise_plan':
                        console.print("[bold red]Analysis concluded that the plan is flawed. Requesting a new plan.[/bold red]")
                        
                        revision_prompt = (
                            f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{original_user_request}'\n\n"
                            "You are in the middle of a multi-step plan. Your own analysis has determined that the "
                            "original plan is flawed or insufficient. Based on the *entire* execution history so far, "
                            "formulate a new, complete `Thought` object with a better plan to achieve the user's "
                            "original goal.\n\n"
                            "<ExecutionHistory>\n"
                            f"{execution_history}"
                            "</ExecutionHistory>"
                        )

                        revision_task = Task(description=revision_prompt, not_main_task=True, response_format=Thought)
                        revision_agent = copy.copy(agent)
                        revision_agent.enable_thinking_tool = False
                        revision_agent.enable_reasoning_tool = False

                        new_thought: Thought = await revision_agent.do_async(revision_task)
                        
                        console.print("[bold magenta]Orchestrator:[/bold magenta] Received revised plan. Restarting execution.")
                        pending_plan = new_thought.plan
                        program_counter = 0
                        execution_history += f"\n--- PLAN REVISED ---\nNew Reasoning: {new_thought.reasoning}\n"
                        continue
                
                program_counter += 1

            console.print("[bold magenta]Orchestrator:[/bold magenta] Plan complete. Preparing for final synthesis.")
            spacing()

            synthesis_prompt = (
                f"Original user request(This is just for remembrance. You have to follow instructions below based on this. But this is not the main focus you will try to fulfill right now): '{original_user_request}'\n\n"
                "You are in the final step of a multi-step task. "
                "You have already executed a plan and gathered all necessary information. "
                "Based *only* on the execution history provided below, synthesize a complete "
                "and final answer for the user's original request.\n\n"
                "<ExecutionHistory>\n"
                f"{execution_history}"
                "</ExecutionHistory>"
            )
            
            synthesis_task = Task(description=synthesis_prompt, not_main_task=True)
            synthesis_agent = copy.copy(agent)
            synthesis_agent.enable_thinking_tool = False
            synthesis_agent.enable_reasoning_tool = False
            final_response = await synthesis_agent.do_async(synthesis_task)
            return final_response

        return orchestrator_wrapper

    def normalize_and_process(self, task_tools: List[Any]) -> Generator[Tuple[Callable, Any], None, None]:
        """
        Processes a list of raw tools from a Task.
        This method iterates through functions, agent instances, and other object methods,
        validates them, and yields a standardized tuple. It also identifies, processes,
        and separates MCP server tools.
        Args:
            task_tools: The raw list of tools from `task.tools`. This list will be
                        modified in place to remove the processed MCP tools.
        Yields:
            A tuple of two possible forms:
            - For a regular tool: (callable_function, ToolConfig)
            - For an MCP tool: (None, mcp_server_instance)
        """
        if self.agent_tool and getattr(self.agent_tool, 'enable_thinking_tool', False):
            setattr(plan_and_execute, '_is_orchestrator', True)
            self._validate_function(plan_and_execute)
            yield (plan_and_execute, ToolConfig())

        from upsonic.agent.agent import Direct
        if not task_tools:
            return
        mcp_tools_to_remove = []
        for tool_item in task_tools:
            if inspect.isclass(tool_item):
                is_mcp_tool = False
                if hasattr(tool_item, 'url'):
                    url = getattr(tool_item, 'url')
                    the_mcp_server = MCPServerSSE(url)
                    yield (None, the_mcp_server)
                    is_mcp_tool = True
                elif hasattr(tool_item, 'command'):
                    env = getattr(tool_item, 'env', {}) if hasattr(tool_item, 'env') and isinstance(getattr(tool_item, 'env', None), dict) else {}
                    command = getattr(tool_item, 'command', None)
                    args = getattr(tool_item, 'args', [])
                    the_mcp_server = MCPServerStdio(command, args=args, env=env)
                    yield (None, the_mcp_server)
                    is_mcp_tool = True
                if is_mcp_tool:
                    mcp_tools_to_remove.append(tool_item)
                    continue
            if inspect.isfunction(tool_item):
                self._validate_function(tool_item)
                config = getattr(tool_item, '_upsonic_tool_config', ToolConfig())
                yield (tool_item, config)
            elif isinstance(tool_item, Direct):
                class_name_base = tool_item.name or f"AgentTool{tool_item.agent_id[:8]}"
                dynamic_class_name = "".join(word.capitalize() for word in re.sub(r"[^a-zA-Z0-9 ]", "", class_name_base).split())
                method_name_base = tool_item.name or f"AgentTool{tool_item.agent_id[:8]}"
                dynamic_method_name = "ask_" + re.sub(r"[^a-zA-Z0-9_]", "", method_name_base.lower().replace(" ", "_"))
                agent_specialty = tool_item.system_prompt or tool_item.company_description or f"a general purpose assistant named '{tool_item.name}'"
                dynamic_docstring = (
                    f"Delegates a sub-task to a specialist agent named '{tool_item.name}'. "
                    f"This agent's role is: {agent_specialty}. "
                    f"Use this tool ONLY for tasks that fall squarely within this agent's described expertise. "
                    f"The 'request' parameter must be a full, clear, and self-contained instruction for the specialist agent."
                )
                async def agent_method_logic(self, request: str) -> str:
                    """This docstring will be replaced dynamically."""
                    the_task = Task(description=request)
                    response = await self.agent.do_async(the_task)
                    return str(response) if response is not None else "The specialist agent returned no response."
                agent_method_logic.__doc__ = dynamic_docstring
                agent_method_logic.__name__ = dynamic_method_name
                def agent_tool_init(self, agent: Direct):
                    self.agent = agent
                AgentToolWrapper = type(
                    dynamic_class_name,
                    (object,),
                    {
                        "__init__": agent_tool_init,
                        dynamic_method_name: agent_method_logic,
                    },
                )
                wrapper_instance = AgentToolWrapper(agent=tool_item)
                for name, method in inspect.getmembers(wrapper_instance, inspect.ismethod):
                    if not name.startswith('_'):
                        self._validate_function(method)
                        config = getattr(method, '_upsonic_tool_config', ToolConfig())
                        yield (method, config)
            elif not inspect.isfunction(tool_item) and hasattr(tool_item, '__class__'):
                is_toolkit = isinstance(tool_item, ToolKit)
                if is_toolkit:
                    for name, method in inspect.getmembers(tool_item, inspect.ismethod):
                        if hasattr(method, '_upsonic_tool_config'):
                            self._validate_function(method)
                            config = getattr(method, '_upsonic_tool_config', ToolConfig())
                            yield (method, config)
                else:
                    for name, method in inspect.getmembers(tool_item, inspect.ismethod):
                        if not name.startswith('_'):
                            self._validate_function(method)
                            config = getattr(method, '_upsonic_tool_config', ToolConfig())
                            yield (method, config)

        if mcp_tools_to_remove:
            for tool in mcp_tools_to_remove:
                while tool in task_tools:
                    task_tools.remove(tool)

    def generate_behavioral_wrapper(self, original_func: Callable, config: ToolConfig) -> Callable:
        """
        Dynamically generates and returns a new function that wraps the original tool.
        This new function contains all the behavioral logic (caching, confirmation, etc.)
        defined in the ToolConfig.
        """
        @functools.wraps(original_func)
        async def behavioral_wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.agent_tool and self.agent_tool.tool_call_limit is not None:
                if self.agent_tool.tool_call_count >= self.agent_tool.tool_call_limit:
                    message = f"Tool call limit of {self.agent_tool.tool_call_limit} has been reached. Cannot execute '{original_func.__name__}'."
                    console.print(f"[bold red]LIMIT REACHED:[/bold red] {message}")
                    spacing()
                    return message
                self.agent_tool.tool_call_count += 1
            func_dict: Dict[str, Any] = {}
            if config.tool_hooks and config.tool_hooks.before:
                result_before = config.tool_hooks.before(*args, **kwargs)
                func_dict["func_before"] = result_before if result_before else None

            if config.requires_confirmation:
                console.print(f"[bold yellow]⚠️ Confirmation Required[/bold yellow]")
                console.print(f"About to execute tool: [cyan]{original_func.__name__}[/cyan]")
                console.print(f"With arguments: [dim]{args}, {kwargs}[/dim]")
                try:
                    confirm = input("Do you want to proceed? (y/n): ").lower().strip()
                except KeyboardInterrupt:
                    confirm = 'n'
                    print()
                if confirm not in ['y', 'yes']:
                    console.print("[bold red]Tool execution cancelled by user.[/bold red]")
                    spacing()
                    return "Tool execution was cancelled by the user."
                spacing()

            if config.requires_user_input and config.user_input_fields:
                console.print(f"[bold blue]📝 User Input Required for {original_func.__name__}[/bold blue]")
                for field_name in config.user_input_fields:
                    try:
                        user_provided_value = input(f"Please provide a value for '{field_name}': ")
                        kwargs[field_name] = user_provided_value
                    except KeyboardInterrupt:
                        console.print("\n[bold red]Input cancelled by user.[/bold red]")
                        return "Tool execution was cancelled by the user during input."
                spacing()

            cache_file = None
            if config.cache_results:
                cache_dir_path = Path(config.cache_dir) if config.cache_dir else Path.home() / '.upsonic' / 'cache'
                arg_string = json.dumps((args, kwargs), sort_keys=True, default=str)
                call_signature = f"{original_func.__name__}:{arg_string}".encode('utf-8')
                cache_key = hashlib.sha256(call_signature).hexdigest()
                cache_file = cache_dir_path / f"{cache_key}.json"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                        is_expired = False
                        if config.cache_ttl is not None:
                            age = time.time() - cache_data.get('timestamp', 0)
                            if age > config.cache_ttl:
                                is_expired = True
                        if not is_expired:
                            console.print(f"[bold green]✓ Cache Hit[/bold green] for tool [cyan]{original_func.__name__}[/cyan]. Returning cached result.")
                            spacing()
                            return cache_data['result']
                        else:
                            cache_file.unlink()
                    except (json.JSONDecodeError, KeyError, OSError):
                        pass

            try:
                if inspect.iscoroutinefunction(original_func):
                    result = await original_func(*args, **kwargs)
                else:
                    # To avoid blocking the event loop with a long-running sync function,
                    # run it in a thread pool executor * _ *
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        None,
                        functools.partial(original_func, *args, **kwargs)
                    )
                func_dict["func"] = result if result else None
                if result and config.show_result:
                    console.print(f"[bold green]✓ Tool Result[/bold green]: {result}")
                    spacing()
                if result and config.stop_after_tool_call:
                    exit()
            except Exception as e:
                console.print(f"[bold red]An error occurred while executing tool '{original_func.__name__}': {e}[/bold red]")
                raise

            if config.cache_results and cache_file:
                try:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_data_to_write = {'timestamp': time.time(), 'result': result}
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data_to_write, f, indent=2, default=str)
                except (TypeError, OSError) as e:
                    console.print(f"[yellow]Warning: Could not cache result for tool '{original_func.__name__}'. Reason: {e}[/yellow]")

            if config.tool_hooks and config.tool_hooks.after:
                result_after = config.tool_hooks.after(result)
                func_dict["funct_after"] = result_after if result_after else None

            setattr(behavioral_wrapper, '_upsonic_stop_after_call', config.stop_after_tool_call)
            setattr(behavioral_wrapper, '_upsonic_show_result', config.show_result)

            return func_dict
        return behavioral_wrapper