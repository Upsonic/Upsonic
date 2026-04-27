from __future__ import annotations

from upsonic.tasks.tasks import Task
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.storage import Memory
    from fastmcp import FastMCP

from upsonic.utils.logging_config import get_env_bool_optional

from .coordinator_setup import CoordinatorSetup
from .delegation_manager import DelegationManager
from .context_sharing import ContextSharing
from .task_assignment import TaskAssignment
from .result_combiner import ResultCombiner


class Team:
    """A callable class for multi-agent operations using the Upsonic client.
    
    Supports both Agent and Team instances as entities, enabling
    hierarchical multi-agent workflows with nested teams.
    Streaming (stream/astream) is text-only; no events are yielded.
    """
    
    def __init__(self,
                 entities: Optional[List[Union[Agent, "Team"]]] = None,
                 tasks: Optional[List[Task]] = None,
                 name: Optional[str] = None,
                 role: Optional[str] = None,
                 goal: Optional[str] = None,
                 model: Optional[Any] = None,
                 response_format: Any = str,
                 ask_other_team_members: bool = False,
                 mode: Literal["sequential", "coordinate", "route"] = "sequential",
                 leader: Optional[Agent] = None,
                 router: Optional[Agent] = None,
                 memory: Optional[Memory] = None,
                 skills: Optional[Any] = None,
                 debug: bool = False,
                 debug_level: int = 1,
                 agents: Optional[List[Union[Agent, "Team"]]] = None,
                 print: Optional[bool] = None,
                 ):
        """
        Initialize the Team with entities (Agents and/or nested Teams) and optionally tasks.

        Args:
            entities: List of Agent and/or Team instances to use as team members.
            tasks: List of tasks to execute (optional).
            name: Display name for this team (used as entity ID when nested).
            role: Role description for this team (used in coordinator prompts when nested).
            goal: Goal description for this team (used in coordinator prompts when nested).
            response_format: The response format for the end task (optional).
            model: The model for internally-created leader/router when leader/router not provided.
            ask_other_team_members: A flag to automatically add other agents as tools.
            mode: The operational mode for the team ('sequential', 'coordinate', or 'route').
            leader: Optional Agent to use as coordinator in 'coordinate' mode. If None, one is created from model.
            router: Optional Agent to use as router in 'route' mode. If None, one is created from model.
            memory: Memory instance for team operations.
            debug: Enable debug logging.
            debug_level: Debug level (1 = standard, 2 = detailed). Only used when debug=True
            agents: Backward-compatible alias for entities.
            print: Enable printing for do() (and allow print_do() when not False). If None, do() does not print unless UPSONIC_AGENT_PRINT=true. If False, print_do() also does not print. UPSONIC_AGENT_PRINT=false overrides everything.
        """
        resolved_entities: List[Union[Agent, Team]] = entities if entities is not None else (agents if agents is not None else [])
        if not resolved_entities:
            raise ValueError("'entities' must be provided with at least one member.")
        self.entities: List[Union[Agent, Team]] = resolved_entities
        self.tasks: List[Task] = tasks if isinstance(tasks, list) else [tasks] if tasks is not None else []
        self.name: Optional[str] = name
        self.role: Optional[str] = role
        self.goal: Optional[str] = goal
        self.model: Optional[Any] = model
        self.response_format: Any = response_format
        self.ask_other_team_members: bool = ask_other_team_members
        self.mode: Literal["sequential", "coordinate", "route"] = mode
        self._leader: Optional[Agent] = leader
        self._router: Optional[Agent] = router
        self.memory: Optional[Memory] = memory
        self.skills: Optional[Any] = skills
        self.debug: bool = debug
        self.debug_level: int = debug_level if debug else 1

        self._print_env: Optional[bool] = get_env_bool_optional("UPSONIC_AGENT_PRINT")
        self._print_param: Optional[bool] = print
        self.print: bool = self._print_env if self._print_env is not None else (print if print is not None else False)

        self.leader_agent: Optional[Agent] = None

        if self.memory:
            self._propagate_memory(self.entities, self.memory)

        if self.skills:
            self._propagate_skills(self.entities, self.skills)

        if self.ask_other_team_members:
            self.add_tool()

    @property
    def agents(self) -> List[Union[Agent, "Team"]]:
        """Backward-compatible alias for entities."""
        return self.entities

    @agents.setter
    def agents(self, value: List[Union[Agent, "Team"]]) -> None:
        """Backward-compatible setter for entities."""
        self.entities = value

    def _propagate_memory(self, entities: List[Union[Agent, "Team"]], memory: "Memory") -> None:
        """Recursively propagate memory to all Agent entities, including those nested in sub-Teams."""
        from upsonic.agent.agent import Agent as AgentClass
        for entity in entities:
            if isinstance(entity, AgentClass):
                if entity.memory is None:
                    entity.memory = memory
            elif isinstance(entity, Team):
                if entity.memory is None:
                    entity.memory = memory
                self._propagate_memory(entity.entities, memory)

    def _propagate_skills(self, entities: List[Union[Agent, "Team"]], skills: Any) -> None:
        """Recursively propagate skills to all Agent entities, including those nested in sub-Teams.

        Each agent receives its own copy of the skills so that metrics are
        tracked independently per agent.  Uses ``agent.add_tools()`` so skill
        tools are properly registered with the agent's ToolManager.
        """
        from upsonic.agent.agent import Agent as AgentClass
        for entity in entities:
            if isinstance(entity, AgentClass):
                if entity.skills is None:
                    # Each agent gets its own copy so metrics are independent
                    agent_skills = skills.copy()
                    entity.skills = agent_skills
                    entity.add_tools(agent_skills.get_tools())
                else:
                    from upsonic.skills import Skills
                    # Remove old skill tools before merging
                    old_skill_tool_names = [
                        t.__name__ for t in entity.tools
                        if hasattr(t, '__name__') and t.__name__.startswith('get_skill_')
                    ]
                    if old_skill_tool_names:
                        entity.remove_tools(old_skill_tool_names)
                    entity.skills = Skills.merge(skills, entity.skills)
                    entity.add_tools(entity.skills.get_tools())
            elif isinstance(entity, Team):
                if entity.skills is None:
                    entity.skills = skills
                self._propagate_skills(entity.entities, skills)

    def get_entity_id(self) -> str:
        """Get display-friendly entity ID for this team."""
        if self.name:
            return self.name
        return f"Team_{id(self)}"

    def _resolve_print_flag(self, method_default: bool) -> bool:
        """Resolve print flag: ENV (UPSONIC_AGENT_PRINT) > constructor print > method default (print_do=True, do=False)."""
        if self._print_env is not None:
            return self._print_env
        if self._print_param is not None:
            return self._print_param
        return method_default

    def _find_first_model(self) -> Optional[Any]:
        """Traverse entities to find the first available model from an Agent."""
        from upsonic.agent.agent import Agent as AgentClass
        for entity in self.entities:
            if isinstance(entity, AgentClass) and hasattr(entity, 'model') and entity.model:
                return entity.model
            if isinstance(entity, Team) and entity.model:
                return entity.model
        for entity in self.entities:
            if isinstance(entity, Team):
                nested_model = entity._find_first_model()
                if nested_model:
                    return nested_model
        return None

    def _format_stream_header(self, entity_id: str, entity_type: str) -> str:
        """Build a visual header line that separates one entity's output from the next."""
        return f"\n\n--- [{entity_type}] {entity_id} ---\n\n"

    async def _entity_astream(
        self,
        entity: Union[Agent, "Team"],
        task: Task,
        *,
        debug: bool = False,
    ) -> AsyncIterator[str]:
        """Stream text from an entity (Agent or Team). Yields str chunks only."""
        from upsonic.agent.agent import Agent as AgentClass
        if isinstance(entity, AgentClass):
            async for chunk in entity.astream(task, events=False, debug=debug):
                if isinstance(chunk, str):
                    yield chunk
        elif hasattr(entity, "astream"):
            async for chunk in entity.astream(task, debug=debug):
                yield chunk
        else:
            result: Any = await entity.do_async(task)
            if result is not None:
                yield str(result)

    def complete(self, tasks: Optional[Union[List[Task], Task]] = None) -> Any:
        return self.do(tasks)
    
    def print_complete(self, tasks: Optional[Union[List[Task], Task]] = None) -> Any:
        return self.print_do(tasks)

    def do(self, tasks: Optional[Union[List[Task], Task]] = None, _print_method_default: Optional[bool] = None) -> Any:
        """
        Execute multi-agent operations with the predefined entities and tasks.
        Whether output is printed follows the same hierarchy as Agent: UPSONIC_AGENT_PRINT env > Team(print=...) > method (do=False).
        
        Args:
            tasks: Optional list of tasks or single task to execute. If not provided, uses tasks from initialization.
            _print_method_default: Internal - when None, uses resolved print from env/constructor (False for do()).
        
        Returns:
            The response from the multi-agent operation
        """
        tasks_to_execute = tasks if tasks is not None else self.tasks
        if not isinstance(tasks_to_execute, list):
            tasks_to_execute = [tasks_to_execute]
        resolved_print: bool = self._resolve_print_flag(False) if _print_method_default is None else _print_method_default
        return self._run_sync(self.multi_agent_async(self.entities, tasks_to_execute, _print_method_default=resolved_print))
    
    def _run_sync(self, coro: Any) -> Any:
        """Run an async coroutine synchronously, handling existing event loops."""
        import asyncio
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            pass
        return asyncio.run(coro)

    async def do_async(
        self,
        task: Union[str, Task],
        *,
        _print_method_default: Optional[bool] = None,
    ) -> Any:
        """
        Execute the team workflow asynchronously for a single task or string.
        When _print_method_default is None, uses resolved print from env/constructor (same as Agent).

        Args:
            task: A Task object or string description to execute.
            _print_method_default: Internal - when None, uses resolved print (False for do_async).

        Returns:
            The response from the multi-agent operation.
        """
        from upsonic.tasks.tasks import Task as TaskClass
        if isinstance(task, str):
            task = TaskClass(description=task)
        tasks_to_execute: List[Task] = [task]
        resolved_print: bool = self._resolve_print_flag(False) if _print_method_default is None else _print_method_default
        result = await self.multi_agent_async(
            self.entities, tasks_to_execute, _print_method_default=resolved_print
        )
        if isinstance(task, TaskClass) and task.response is None and result is not None:
            task._response = str(result) if not isinstance(result, str) else result
        return result

    async def print_do_async(self, task: Union[str, Task]) -> Any:
        """Execute the team workflow asynchronously with print output. Respects UPSONIC_AGENT_PRINT and Team(print=...)."""
        resolved_print: bool = self._resolve_print_flag(True)
        return await self.do_async(task, _print_method_default=resolved_print)

    async def ado(
        self,
        task: Union[str, Task],
        *,
        _print_method_default: Optional[bool] = None,
    ) -> Any:
        """Async alias for do_async."""
        return await self.do_async(task, _print_method_default=_print_method_default)

    async def multi_agent_async(self, entity_configurations: List[Union[Agent, Team]], tasks: Any, _print_method_default: bool = False) -> Any:
        """
        Asynchronous multi-entity execution engine.
        
        Args:
            entity_configurations: List of Agent and/or Team instances to use as team members.
            tasks: Tasks to execute.
            _print_method_default: Internal - default print value based on method (do=False, print_do=True)
        """
        from upsonic.agent.agent import Agent as AgentClass

        if not isinstance(tasks, list):
            tasks = [tasks]

        if self.mode == "sequential":
            context_sharing = ContextSharing()
            task_assignment = TaskAssignment()
            combiner_model = self.model
            if not combiner_model:
                combiner_model = self._find_first_model()

            last_debug: bool = False
            for entity in reversed(self.entities):
                if isinstance(entity, AgentClass) and hasattr(entity, 'debug'):
                    last_debug = entity.debug
                    break

            result_combiner = ResultCombiner(model=combiner_model, debug=last_debug)
            entities_registry, entity_names = task_assignment.prepare_entities_registry(entity_configurations)
            all_results: List[Task] = []
            for task_index, current_task in enumerate(tasks):
                selection_context = context_sharing.build_selection_context(
                    current_task, tasks, task_index, entity_configurations, all_results
                )
                selected_entity_name = await task_assignment.select_entity_for_task(
                    current_task, selection_context, entities_registry, entity_names, entity_configurations
                )
                
                if self.debug and self.debug_level >= 2:
                    from upsonic.utils.printing import debug_log_level2
                    debug_log_level2(
                        f"Team task assignment (task {task_index + 1}/{len(tasks)})",
                        "Team",
                        debug=self.debug,
                        debug_level=self.debug_level,
                        task_index=task_index,
                        total_tasks=len(tasks),
                        task_description=current_task.description[:300] if hasattr(current_task, 'description') else None,
                        selected_agent=selected_entity_name,
                        available_agents=entity_names,
                        context_keys=list(selection_context.keys()) if isinstance(selection_context, dict) else None,
                        previous_results_count=len(all_results)
                    )
                
                if selected_entity_name:
                    context_sharing.enhance_task_context(
                        current_task, tasks, task_index, entity_configurations, all_results
                    )
                    selected_entity = entities_registry[selected_entity_name]
                    if self.debug:
                        _etype: str = "Team" if hasattr(selected_entity, "entities") else "Agent"
                        from upsonic.utils.printing import info_log
                        info_log(
                            f"Sequential mode — Calling {_etype} '{selected_entity_name}' "
                            f"(task {task_index + 1}/{len(tasks)}) | "
                            f"Description: {current_task.description[:200]}",
                            context="Team",
                        )
                    result = await selected_entity.do_async(current_task, _print_method_default=_print_method_default)
                    all_results.append(current_task)
                    
                    if self.debug and self.debug_level >= 2:
                        from upsonic.utils.printing import debug_log_level2
                        debug_log_level2(
                            f"Team task completed (task {task_index + 1}/{len(tasks)})",
                            "Team",
                            debug=self.debug,
                            debug_level=self.debug_level,
                            task_index=task_index,
                            agent_name=selected_entity_name,
                            result_preview=str(result)[:500] if result else None,
                            task_duration=getattr(current_task, 'duration', None),
                            task_cost=getattr(current_task, 'total_cost', None)
                        )
            if not result_combiner.should_combine_results(all_results):
                return result_combiner.get_single_result(all_results)
            return await result_combiner.combine_results(
                all_results, self.response_format, self.entities
            )

        elif self.mode == "coordinate":
            if self._leader is None and not self.model:
                raise ValueError(f"For '{self.mode}' mode either pass a `leader` Agent or set `model` on the Team.")
            tool_mapping: Dict[str, Any] = {}
            for task in self.tasks:
                if task.tools:
                    for tool in task.tools:
                        if callable(tool):
                            tool_mapping[tool.__name__] = tool

            setup_manager = CoordinatorSetup(self.entities, tasks, mode="coordinate")
            delegation_manager = DelegationManager(self.entities, tool_mapping, debug=self.debug)

            if self._leader is not None:
                self.leader_agent = self._leader
                if self.leader_agent.memory is None and self.memory is not None:
                    self.leader_agent.memory = self.memory
            else:
                self.leader_agent = AgentClass(
                    model=self.model,
                    memory=self.memory,
                )

            leader_system_prompt: str = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt

            master_description: str = (
                "Begin your mission. Review your system prompt for the full list of tasks and your team roster. "
                "Formulate your plan and start delegating tasks now."
            )

            all_attachments: List[str] = []
            for task in tasks:
                if task.attachments:
                    all_attachments.extend(task.attachments)

            delegation_tool = delegation_manager.get_delegation_tool()

            master_task: Task = Task(
                description=master_description,
                attachments=all_attachments if all_attachments else None,
                tools=[delegation_tool],
                response_format=self.response_format,
            )

            final_response = await self.leader_agent.do_async(master_task, _print_method_default=_print_method_default)
            
            return final_response
        elif self.mode == "route":
            if self._router is None and not self.model:
                raise ValueError(f"For '{self.mode}' mode either pass a `router` Agent or set `model` on the Team.")

            setup_manager = CoordinatorSetup(self.entities, tasks, mode="route")
            delegation_manager = DelegationManager(self.entities, {}, debug=self.debug)

            if self._router is not None:
                self.leader_agent = self._router
            else:
                self.leader_agent = AgentClass(model=self.model)

            leader_system_prompt = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt
            routing_tool = delegation_manager.get_routing_tool()

            router_task_description: str = "Analyze the MISSION OBJECTIVES in your system prompt and route the request to the best specialist."
            router_task: Task = Task(description=router_task_description, tools=[routing_tool])

            await self.leader_agent.do_async(router_task, _print_method_default=_print_method_default)

            chosen_entity = delegation_manager.routed_entity

            if not chosen_entity:
                raise ValueError("Routing failed: The router agent did not select a team member.")
            
            consolidated_description: str = " ".join([task.description for task in tasks])
            all_attachments = [attachment for task in tasks if task.attachments for attachment in task.attachments]
            all_tools = [tool for task in tasks if task.tools for tool in task.tools]

            final_task: Task = Task(
                description=consolidated_description,
                attachments=all_attachments or None,
                tools=list(set[Any](all_tools)) if all_tools else None,
                response_format=self.response_format
            )

            if self.debug:
                _chosen_id: str = chosen_entity.get_entity_id()
                _etype: str = "Team" if hasattr(chosen_entity, "entities") else "Agent"
                from upsonic.utils.printing import info_log
                info_log(
                    f"Route mode — Routed to {_etype} '{_chosen_id}' | "
                    f"Description: {consolidated_description[:200]} | "
                    f"Tools: {[t.__name__ if hasattr(t, '__name__') else str(t) for t in all_tools] if all_tools else None} | "
                    f"Attachments: {all_attachments or None}",
                    context="Team",
                )

            await chosen_entity.do_async(final_task, _print_method_default=_print_method_default)
            return final_task.response

    def print_do(self, tasks: Optional[Union[List[Task], Task]] = None) -> Any:
        """
        Execute the multi-agent operation and print the result.
        Respects UPSONIC_AGENT_PRINT and Team(print=...) (e.g. Team(print=False).print_do() does not print).
        
        Returns:
            The response from the multi-agent operation
        """
        resolved_print: bool = self._resolve_print_flag(True)
        return self.do(tasks, _print_method_default=resolved_print)

    async def astream(
        self,
        tasks: Optional[Union[List[Task], Task, str]] = None,
        *,
        debug: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream task execution asynchronously. Yields text chunks only (no events).
        In coordinate mode, only the leader's output is streamed; delegated member outputs are not streamed.

        Args:
            tasks: Task, list of Tasks, string description, or None (falls back to self.tasks).
            debug: Enable debug mode.

        Yields:
            str: Text chunks as they arrive.
        """
        from upsonic.tasks.tasks import Task as TaskClass
        from upsonic.agent.agent import Agent as AgentClass
        tasks_to_execute: Optional[Union[List[Task], Task, str]] = tasks if tasks is not None else self.tasks
        if isinstance(tasks_to_execute, str):
            tasks_to_execute = [TaskClass(description=tasks_to_execute)]
        elif isinstance(tasks_to_execute, Task):
            tasks_to_execute = [tasks_to_execute]
        elif not isinstance(tasks_to_execute, list):
            tasks_to_execute = [tasks_to_execute]
        tasks_list: List[Task] = tasks_to_execute
        mode_debug: bool = debug or self.debug

        if self.mode == "sequential":
            context_sharing = ContextSharing()
            task_assignment = TaskAssignment()
            combiner_model: Optional[Any] = self.model or self._find_first_model()
            last_debug = False
            for entity in reversed(self.entities):
                if isinstance(entity, AgentClass) and hasattr(entity, "debug"):
                    last_debug = entity.debug
                    break
            result_combiner = ResultCombiner(model=combiner_model, debug=last_debug)
            entities_registry, entity_names = task_assignment.prepare_entities_registry(self.entities)
            all_results: List[Task] = []
            for task_index, current_task in enumerate(tasks_list):
                selection_context = context_sharing.build_selection_context(
                    current_task, tasks_list, task_index, self.entities, all_results
                )
                selected_entity_name = await task_assignment.select_entity_for_task(
                    current_task, selection_context, entities_registry, entity_names, self.entities
                )
                if selected_entity_name:
                    context_sharing.enhance_task_context(
                        current_task, tasks_list, task_index, self.entities, all_results
                    )
                    selected_entity = entities_registry[selected_entity_name]
                    entity_type: str = "Team" if isinstance(selected_entity, Team) else "Agent"
                    yield self._format_stream_header(selected_entity_name, entity_type)
                    async for chunk in self._entity_astream(selected_entity, current_task, debug=mode_debug):
                        yield chunk
                    all_results.append(current_task)
            if result_combiner.should_combine_results(all_results):
                combined: Any = await result_combiner.combine_results(
                    all_results, self.response_format, self.entities
                )
                if combined is not None:
                    yield str(combined)

        elif self.mode == "coordinate":
            if self._leader is None and not self.model:
                raise ValueError(f"For '{self.mode}' mode either pass a `leader` Agent or set `model` on the Team.")
            tool_mapping: Dict[str, Any] = {}
            for t in self.tasks:
                if t.tools:
                    for tool in t.tools:
                        if callable(tool):
                            tool_mapping[tool.__name__] = tool
            setup_manager = CoordinatorSetup(self.entities, tasks_list, mode="coordinate")
            delegation_manager = DelegationManager(self.entities, tool_mapping, debug=self.debug)
            if self._leader is not None:
                self.leader_agent = self._leader
                if self.leader_agent.memory is None and self.memory is not None:
                    self.leader_agent.memory = self.memory
            else:
                self.leader_agent = AgentClass(model=self.model, memory=self.memory)
            leader_system_prompt = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt
            master_description = (
                "Begin your mission. Review your system prompt for the full list of tasks and your team roster. "
                "Formulate your plan and start delegating tasks now."
            )
            all_attachments = [a for t in tasks_list if t.attachments for a in t.attachments]
            delegation_tool = delegation_manager.get_delegation_tool()
            master_task = Task(
                description=master_description,
                attachments=all_attachments or None,
                tools=[delegation_tool],
                response_format=self.response_format,
            )
            leader_id: str = self.leader_agent.name or self.leader_agent.get_entity_id() if hasattr(self.leader_agent, "get_entity_id") else "Leader"
            yield self._format_stream_header(leader_id, "Leader")
            async for chunk in self.leader_agent.astream(master_task, events=False, debug=mode_debug):
                if isinstance(chunk, str):
                    yield chunk

        elif self.mode == "route":
            if self._router is None and not self.model:
                raise ValueError(f"For '{self.mode}' mode either pass a `router` Agent or set `model` on the Team.")
            setup_manager = CoordinatorSetup(self.entities, tasks_list, mode="route")
            delegation_manager = DelegationManager(self.entities, {}, debug=self.debug)
            if self._router is not None:
                self.leader_agent = self._router
            else:
                self.leader_agent = AgentClass(model=self.model)
            leader_system_prompt = setup_manager.create_leader_prompt()
            self.leader_agent.system_prompt = leader_system_prompt
            routing_tool = delegation_manager.get_routing_tool()
            router_task = Task(
                description="Analyze the MISSION OBJECTIVES in your system prompt and route the request to the best specialist.",
                tools=[routing_tool],
            )
            await self.leader_agent.do_async(router_task)
            chosen_entity = delegation_manager.routed_entity
            if not chosen_entity:
                raise ValueError("Routing failed: The router agent did not select a team member.")
            consolidated_description = " ".join([t.description for t in tasks_list])
            all_attachments = [a for t in tasks_list if t.attachments for a in t.attachments]
            all_tools = [tool for t in tasks_list if t.tools for tool in t.tools]
            final_task = Task(
                description=consolidated_description,
                attachments=all_attachments or None,
                tools=list(set(all_tools)) if all_tools else None,
                response_format=self.response_format,
            )
            chosen_id: str = chosen_entity.get_entity_id()
            chosen_type: str = "Team" if isinstance(chosen_entity, Team) else "Agent"
            yield self._format_stream_header(chosen_id, chosen_type)
            async for chunk in self._entity_astream(chosen_entity, final_task, debug=mode_debug):
                yield chunk

    def stream(
        self,
        tasks: Optional[Union[List[Task], Task, str]] = None,
        *,
        debug: bool = False,
    ) -> Iterator[str]:
        """
        Stream task execution synchronously. Yields text chunks only (no events).
        For async streaming use astream() instead.

        Args:
            tasks: Task, list of Tasks, string description, or None (falls back to self.tasks).
            debug: Enable debug mode.

        Yields:
            str: Text chunks as they arrive.
        """
        import asyncio
        import queue
        import threading
        result_queue: queue.Queue = queue.Queue()
        error_holder: List[Exception] = []

        async def stream_to_queue() -> None:
            try:
                async for chunk in self.astream(tasks, debug=debug):
                    result_queue.put(chunk)
            except Exception as e:
                error_holder.append(e)
            finally:
                result_queue.put(None)

        def run_async_stream() -> None:
            asyncio.run(stream_to_queue())

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        thread = threading.Thread(target=run_async_stream, daemon=True)
        thread.start()
        while True:
            item = result_queue.get()
            if item is None:
                if error_holder:
                    raise error_holder[0]
                break
            yield item

    def as_mcp(self, name: Optional[str] = None) -> "FastMCP":
        """
        Expose this team as an MCP server.

        Creates a FastMCP server with a ``do`` tool that delegates task
        execution to this team's multi-agent workflow.  The returned server
        can be started with any transport (stdio, sse, streamable-http) via
        its ``.run()`` method.

        Args:
            name: MCP server name. Defaults to the team's name or
                  ``"Upsonic Team"``.

        Returns:
            A :class:`fastmcp.FastMCP` server instance ready to ``.run()``.
        """
        try:
            from fastmcp import FastMCP as _FastMCP  # type: ignore[import-not-found]
        except ImportError:
            from upsonic.utils.printing import import_error
            import_error(
                package_name="fastmcp",
                install_command="pip install 'upsonic[mcp]'",
                feature_name="Team.as_mcp() (expose team as an MCP server)",
            )

        server_name: str = name or self.name or "Upsonic Team"
        server: _FastMCP = _FastMCP(server_name)

        description_parts: List[str] = []
        if self.role:
            description_parts.append(f"Role: {self.role}")
        if self.goal:
            description_parts.append(f"Goal: {self.goal}")

        entity_names: List[str] = []
        from upsonic.agent.agent import Agent as AgentClass
        for entity in self.entities:
            if isinstance(entity, AgentClass):
                entity_names.append(entity.name or "Agent")
            elif isinstance(entity, Team):
                entity_names.append(entity.name or "Team")
        if entity_names:
            description_parts.append(f"Members: {', '.join(entity_names)}")

        description_parts.append(f"Mode: {self.mode}")

        tool_description: str = f"Execute a task using the {server_name} team."
        if description_parts:
            tool_description += " " + " | ".join(description_parts)

        team_ref: "Team" = self

        @server.tool(description=tool_description)
        def do(task: str) -> str:
            """Give a task to this team and get the result."""
            task_obj: Task = Task(description=task, response_format=team_ref.response_format)
            result: Any = team_ref.print_do(tasks=task_obj)
            if result is None:
                return ""
            return str(result)

        return server

    def add_tool(self) -> None:
        """
        Add Agent entities as tools to each Task object.
        Only Agent instances are added; Team instances are skipped.
        """
        from upsonic.agent.agent import Agent as AgentClass
        agent_entities = [e for e in self.entities if isinstance(e, AgentClass)]
        for task in self.tasks:
            if not hasattr(task, 'tools'):
                task.tools = []
            if isinstance(task.tools, list):
                task.tools.extend(agent_entities)
            else:
                task.tools = agent_entities
