"""
StateGraph - The core graph execution engine.

This module provides the StateGraph class which enables building stateful,
multi-step workflows with LLMs using a graph-based approach.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

if TYPE_CHECKING:
    from upsonic.uel.graph import RunnableGraph

from upsonic.graphv2.cache import BaseCache, CachePolicy
from upsonic.graphv2.checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    StateSnapshot,
    generate_checkpoint_id,
)
from upsonic.graphv2.errors import (
    GraphExecutionUncertainError,
    GraphRecursionError,
    GraphValidationError,
)
from upsonic.graphv2.primitives import END, Command, InterruptException, Send
from upsonic.graphv2.store import BaseStore
from upsonic.graphv2.task import RetryPolicy
from upsonic.uel.runnable import Runnable

# Special START marker
START = "__start__"


# Type for state classes
StateT = TypeVar('StateT', bound=Dict[str, Any])


@dataclass
class NodeConfig:
    """Configuration for a single node in the graph.
    
    Attributes:
        func: The node function to execute
        retry_policy: Optional retry configuration for failures
        cache_policy: Optional cache configuration for results
    """
    
    func: Callable[..., Any]
    retry_policy: Optional[RetryPolicy] = None
    cache_policy: Optional[CachePolicy] = None
    retry_safe: bool = False

    def __post_init__(self) -> None:
        if type(self.retry_safe) is not bool:
            raise TypeError("retry_safe must be a boolean")


@dataclass
class Edge:
    """A connection between two nodes in the graph.
    
    Attributes:
        from_node: Source node name
        to_node: Target node name
    """
    
    from_node: str
    to_node: str


@dataclass
class ConditionalEdge:
    """A conditional edge that routes based on state.
    
    Attributes:
        from_node: Source node name
        condition: Function that returns the target node name
        targets: List of possible target node names
    """
    
    from_node: str
    condition: Callable[[Dict[str, Any]], str]
    targets: List[str]


class StateGraph(Runnable[Dict[str, Any], Dict[str, Any]]):
    """A stateful graph for building multi-step LLM workflows.
    
    StateGraph allows you to define a workflow as a graph where:
    - Nodes represent computation steps (functions that take state and return updates)
    - Edges define how control flows between nodes
    - State is shared across all nodes and persisted across executions
    - Interrupts enable human-in-the-loop workflows
    - Checkpointing provides durability and time travel
    
    The graph inherits from Runnable, making it compatible with uel chains.
    
    Example:
        ```python
        from typing_extensions import TypedDict
        from upsonic.graphv2 import StateGraph, START, END
        
        class MyState(TypedDict):
            messages: list[str]
            count: int
        
        def my_node(state: MyState) -> dict:
            return {"count": state["count"] + 1}
        
        builder = StateGraph(MyState)
        builder.add_node("process", my_node)
        builder.add_edge(START, "process")
        builder.add_edge("process", END)
        
        graph = builder.compile()
        result = graph.invoke({"messages": [], "count": 0})
        ```
    """
    
    def __init__(
        self,
        state_schema: Type[StateT],
        *,
        input_schema: Optional[Type] = None,
        output_schema: Optional[Type] = None,
        context_schema: Optional[Type] = None
    ):
        """Initialize the graph builder.
        
        Args:
            state_schema: TypedDict class defining the state structure
            input_schema: Optional schema for input validation
            output_schema: Optional schema for output filtering
            context_schema: Optional schema for runtime context
        """
        self.state_schema = state_schema
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.context_schema = context_schema
        self.nodes: Dict[str, NodeConfig] = {}
        self.edges: List[Edge] = []
        self.conditional_edges: List[ConditionalEdge] = []
        self._reducers: Dict[str, Callable] = {}
        
        # Extract reducers from state schema annotations
        self._extract_reducers()
    
    def _extract_reducers(self):
        """Extract reducers from Annotated types in the state schema."""
        if hasattr(self.state_schema, '__annotations__'):
            for field_name, field_type in self.state_schema.__annotations__.items():
                # Check if this is an Annotated type
                origin = get_origin(field_type)
                if origin is not None:
                    # For Annotated types, get the metadata
                    args = get_args(field_type)
                    if len(args) > 1:
                        # The reducer is in the metadata (second argument)
                        metadata = args[1]
                        if callable(metadata):
                            self._reducers[field_name] = metadata
    
    def add_node(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None,
        retry_safe: bool = False
    ) -> None:
        """Add a node to the graph.
        
        Args:
            name: Unique name for the node
            func: Function that takes state and returns state updates
            retry_policy: Optional retry configuration
            cache_policy: Optional cache configuration
            retry_safe: Explicitly permit replay after an uncertain outcome
            
        Raises:
            ValueError: If a node with this name already exists
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        
        if name in (START, END):
            raise ValueError(f"Cannot use reserved name '{name}' for a node")
        
        self.nodes[name] = NodeConfig(
            func=func,
            retry_policy=retry_policy,
            cache_policy=cache_policy,
            retry_safe=retry_safe
        )
    
    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a normal (unconditional) edge between two nodes.
        
        Args:
            from_node: Source node name (or START)
            to_node: Target node name (or END)
        """
        self.edges.append(Edge(from_node=from_node, to_node=to_node))
    
    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable[[Dict[str, Any]], str],
        targets: List[str]
    ) -> None:
        """Add a conditional edge that routes based on state.
        
        The condition function receives the current state and returns the name
        of the next node to execute.
        
        Args:
            from_node: Source node name
            condition: Function that returns next node name
            targets: List of possible target nodes (for validation)
        """
        self.conditional_edges.append(
            ConditionalEdge(from_node=from_node, condition=condition, targets=targets)
        )
    
    def compile(
        self,
        *,
        checkpointer: Optional[BaseCheckpointer] = None,
        store: Optional[BaseStore] = None,
        cache: Optional[BaseCache] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None,
        durability: Literal["exit", "async", "sync"] = "async"
    ) -> "CompiledStateGraph":
        """Compile the graph into an executable workflow.
        
        Args:
            checkpointer: Optional checkpointer for persistence
            store: Optional store for cross-thread memory
            cache: Optional cache for node-level caching
            interrupt_before: List of nodes to pause before
            interrupt_after: List of nodes to pause after
            durability: Checkpoint writing mode
            
        Returns:
            Compiled graph ready for execution
        """
        # Validate the graph structure
        self._validate_graph()
        
        # Create compiled graph
        compiled = CompiledStateGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            reducers=self._reducers,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            context_schema=self.context_schema,
            checkpointer=checkpointer,
            store=store,
            cache=cache,
            interrupt_before=interrupt_before or [],
            interrupt_after=interrupt_after or [],
            durability=durability,
        )
        
        return compiled
    
    def _validate_graph(self):
        """Validate the graph structure."""
        # Check that all referenced nodes exist
        all_nodes = set(self.nodes.keys())
        
        for edge in self.edges:
            if edge.from_node not in (START, *all_nodes):
                raise ValueError(f"Edge references unknown node: {edge.from_node}")
            if edge.to_node not in (END, *all_nodes):
                raise ValueError(f"Edge references unknown node: {edge.to_node}")
        
        for cond_edge in self.conditional_edges:
            if cond_edge.from_node not in all_nodes:
                raise ValueError(f"Conditional edge references unknown node: {cond_edge.from_node}")
            
            for target in cond_edge.targets:
                if target not in (END, *all_nodes):
                    raise ValueError(f"Conditional edge target not found: {target}")
    
    def get_graph(self) -> "RunnableGraph":
        """Get a graph representation of this state graph for visualization.
        
        Returns a RunnableGraph that can render the graph structure as
        ASCII art or Mermaid diagrams. Normal edges are shown as solid
        connections, conditional edges as dashed connections.
        
        Returns:
            A RunnableGraph object with to_ascii(), to_mermaid(), and
            get_structure_details() methods
        """
        from upsonic.uel.graph import RunnableGraph
        return RunnableGraph(self)
    
    def invoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the graph synchronously (not yet compiled).
        
        This is here for Runnable interface compatibility but should use
        the compiled graph for actual execution.
        """
        raise RuntimeError("Graph must be compiled before execution. Use graph.compile().invoke()")
    
    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the graph asynchronously (not yet compiled)."""
        raise RuntimeError("Graph must be compiled before execution. Use graph.compile().ainvoke()")


class CompiledStateGraph(Runnable[Dict[str, Any], Dict[str, Any]]):
    """A compiled, executable state graph.
    
    This is returned by StateGraph.compile() and provides the actual
    execution engine for the graph.
    """
    
    def __init__(
        self,
        nodes: Dict[str, NodeConfig],
        edges: List[Edge],
        conditional_edges: List[ConditionalEdge],
        reducers: Dict[str, Callable],
        input_schema: Optional[Type],
        output_schema: Optional[Type],
        context_schema: Optional[Type],
        checkpointer: Optional[BaseCheckpointer],
        store: Optional[BaseStore],
        cache: Optional[BaseCache],
        interrupt_before: List[str],
        interrupt_after: List[str],
        durability: str,
    ):
        """Initialize the compiled graph."""
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.reducers = reducers
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.context_schema = context_schema
        self.checkpointer = checkpointer
        self.store = store
        self.cache = cache
        self.interrupt_before = interrupt_before
        self.interrupt_after = interrupt_after
        self.durability = durability
        
        # Build routing maps for efficient execution
        self._build_routing_maps()
    
    def _build_routing_maps(self):
        """Build efficient routing structures."""
        self.normal_routes: Dict[str, List[str]] = {}
        self.conditional_routes: Dict[str, ConditionalEdge] = {}
        
        # Map normal edges - support multiple edges from same node
        for edge in self.edges:
            if edge.from_node not in self.normal_routes:
                self.normal_routes[edge.from_node] = []
            self.normal_routes[edge.from_node].append(edge.to_node)
        
        # Map conditional edges
        for cond_edge in self.conditional_edges:
            self.conditional_routes[cond_edge.from_node] = cond_edge
    
    def invoke(
        self,
        input: Optional[Union[Dict[str, Any], Command]] = None,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the graph synchronously.
        
        Args:
            input: Initial state or Command to resume from interrupt
            config: Configuration including thread_id and checkpoint_id
            context: Runtime context parameters
            
        Returns:
            Final state dictionary
        """
        return asyncio.run(self.ainvoke(input, config, context))
    
    async def ainvoke(
        self,
        input: Optional[Union[Dict[str, Any], Command]] = None,
        config: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute or recover a graph using durable logical invocation records.

        Checkpointer writes are synchronous safety barriers in every durability
        mode. Use a persistent checkpointer for recovery across processes.
        Concurrent writers to the same thread are not supported.
        """
        config = dict(config or {})
        thread_config = config.get("configurable", {})
        thread_id = thread_config.get("thread_id", "default")
        checkpoint_id = thread_config.get("checkpoint_id")
        recursion_limit = config.get("recursion_limit", 100)
        if context:
            config["context"] = context
        if input is not None and not isinstance(input, Command) and self.input_schema:
            input = self._validate_input(input)
        checkpoint = self.checkpointer.get(thread_id, checkpoint_id) if self.checkpointer else None
        if checkpoint_id and self.checkpointer:
            latest = self.checkpointer.get(thread_id)
            if latest and latest.checkpoint_id != checkpoint_id and latest.pending_tasks:
                raise GraphExecutionUncertainError(latest.pending_tasks)
        if isinstance(input, Command):
            try:
                return self._filter_output(await self._resume_from_interrupt(
                    input, thread_id, checkpoint_id, config, recursion_limit))
            except InterruptException as exc:
                latest = self.checkpointer.get(thread_id) if self.checkpointer else None
                return {**(latest.state if latest else {}), "__interrupt__": [{"value": exc.value}]}
        if input is None and self.checkpointer and checkpoint is None:
            raise ValueError(f"Cannot resume: no checkpoint found for thread_id={thread_id}")
        state = deepcopy(checkpoint.state) if checkpoint else deepcopy(input or {})
        next_nodes = checkpoint.next_nodes if checkpoint else [START]
        tasks = deepcopy(checkpoint.pending_tasks) if checkpoint else None
        if checkpoint and input is not None:
            if tasks:
                # Changing inputs must not replace the original invocation plan.
                raise GraphExecutionUncertainError(tasks)
            state = self._merge_state(state, input)
            if END in next_nodes or not next_nodes:
                next_nodes = [START]
                tasks = None
        if (checkpoint and checkpoint.pending_tasks is None
                and checkpoint.next_nodes and END not in checkpoint.next_nodes):
            tasks = self._plan_tasks(next_nodes, state)
            for task in tasks:
                task["status"] = "UNKNOWN"
        try:
            final_state = await self._execute_graph(
                state, next_nodes, thread_id,
                checkpoint.checkpoint_id if checkpoint else None,
                config, recursion_limit, pending_tasks=tasks)
            return self._filter_output(final_state)
        except InterruptException as exc:
            latest = self.checkpointer.get(thread_id) if self.checkpointer else None
            return {**(latest.state if latest else state), "__interrupt__": [{"value": exc.value}]}

    def _validate_input(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input against input_schema.
        
        Args:
            input: Input dictionary to validate
            
        Returns:
            Validated input
            
        Raises:
            GraphValidationError: If input doesn't match schema
        """
        if self.input_schema is None:
            return input
        
        # Check if input has required keys from input_schema
        if hasattr(self.input_schema, '__annotations__'):
            schema_keys = set(self.input_schema.__annotations__.keys())
            input_keys = set(input.keys())
            
            # Check for missing required keys
            missing_keys = schema_keys - input_keys
            if missing_keys:
                raise GraphValidationError(
                    f"Input missing required keys: {missing_keys}"
                )
        
        return input
    
    def _filter_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Filter output to match output_schema.
        
        Args:
            state: Final state dictionary
            
        Returns:
            Filtered output containing only output_schema keys
        """
        if self.output_schema is None:
            return state
        
        # Filter to only output_schema keys
        if hasattr(self.output_schema, '__annotations__'):
            output_keys = set(self.output_schema.__annotations__.keys())
            return {k: v for k, v in state.items() if k in output_keys}
        
        return state
    
    def _plan_tasks(
        self, targets: list[str | Send], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Materialize ordered invocation slots; payload equality is not identity."""
        tasks = []
        for target in targets:
            if target == START:
                tasks.extend(self._plan_tasks(self._get_next_nodes(START, state), state))
                continue
            if target == END:
                continue
            node = target.node if isinstance(target, Send) else target
            if node not in self.nodes:
                raise ValueError(f"Send or edge references unknown node: {node}")
            node_state = {**state, **target.state} if isinstance(target, Send) else state
            tasks.append({"id": generate_checkpoint_id(), "node": node,
                          "state": deepcopy(node_state), "send": isinstance(target, Send),
                          "status": "NOT_STARTED", "result": None})
        return tasks

    async def _execute_graph(
        self,
        state: dict[str, Any],
        next_nodes: list[str],
        thread_id: str,
        parent_checkpoint_id: str | None,
        config: dict[str, Any],
        recursion_limit: int,
        pending_tasks: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Plan, durably admit, and settle each logical invocation in a batch."""
        tasks: list[dict[str, Any]] = pending_tasks if pending_tasks is not None else []
        parent = parent_checkpoint_id

        def save(metadata: dict[str, Any] | None = None) -> None:
            nonlocal parent
            checkpoint = Checkpoint(
                checkpoint_id=generate_checkpoint_id(), thread_id=thread_id,
                state=deepcopy(state), next_nodes=[t["node"] for t in tasks] or [END],
                parent_checkpoint_id=parent, metadata=metadata or {},
                pending_tasks=deepcopy(tasks))
            if self.checkpointer:
                # Never schedule this write in the background or buffer until exit.
                self.checkpointer.put(checkpoint)
            parent = checkpoint.checkpoint_id

        if pending_tasks is None:
            tasks = self._plan_tasks(next_nodes, state)
            save()
        for _ in range(recursion_limit):
            if not tasks:
                return state
            uncertain = [task for task in tasks
                         if task["status"] not in ("NOT_STARTED", "COMPLETED")
                         and not self.nodes[task["node"]].retry_safe]
            if uncertain:
                raise GraphExecutionUncertainError(uncertain)
            for task in tasks:
                if (task["status"] == "NOT_STARTED" and task["node"] in self.interrupt_before
                        and not task.get("before_approved", False)):
                    save({"interrupted_before": task["node"], "interrupted_task": task["id"]})
                    raise InterruptException({"node": task["node"], "when": "before"})

            async def execute(
                task: dict[str, Any]
            ) -> dict[str, Any] | Command:
                if task["status"] == "COMPLETED":
                    return task["result"]
                node_config = self.nodes[task["node"]]
                node_runtime = {**config, "configurable": {
                    **config.get("configurable", {}), "thread_id": thread_id,
                    "node_execution_id": task["id"]}}
                # Cache hits never enter a node body, hence need no dispatch marker.
                if task["status"] == "NOT_STARTED" and node_config.cache_policy and self.cache:
                    cache_key = node_config.cache_policy.key_func(task["state"])
                    cached = self.cache.get((task["node"],), cache_key)
                    if cached is not None:
                        task["result"] = cached
                        task["status"] = "COMPLETED"
                        save()
                        return cached
                task["status"] = "DISPATCH_RECORDED"
                save()  # A failing barrier aborts before any body execution.
                try:
                    from upsonic.graphv2.primitives import (
                        _interrupt_counter,
                        _interrupt_resume_values,
                    )
                    resume_token = counter_token = None
                    values = config.get("_graph_resume_values", {}).get(task["id"])
                    if values is not None:
                        resume_token = _interrupt_resume_values.set(values)
                        counter_token = _interrupt_counter.set(0)
                    try:
                        result = await self._execute_node_with_policies(
                            task["node"], node_config, deepcopy(task["state"]), node_runtime)
                    finally:
                        if resume_token is not None:
                            _interrupt_resume_values.reset(resume_token)
                            _interrupt_counter.reset(counter_token)
                    if isinstance(result, Send) or (isinstance(result, list)
                            and any(isinstance(value, Send) for value in result)):
                        raise ValueError("Send objects can only be returned from conditional edge functions")
                    if not isinstance(result, (dict, Command)):
                        raise TypeError("Node result must be a dictionary or Command")
                except InterruptException as exc:
                    task["status"] = "INTERRUPTED"
                    save({"interrupted_node": task["node"], "interrupted_task": task["id"],
                          "interrupt_value": exc.value})
                    raise
                except Exception as exc:
                    task["status"] = "UNKNOWN"
                    save({"failed_node": task["node"], "error": str(exc)})
                    if not node_config.retry_safe:
                        raise GraphExecutionUncertainError([task]) from exc
                    raise
                task["result"] = result
                task["status"] = "COMPLETED"
                save()  # Persist each sibling receipt before a batch can advance.
                return result

            outcomes = await asyncio.gather(*(execute(task) for task in tasks), return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            completed = tasks
            for task in completed:
                result = task["result"]
                updates = result.update or {} if isinstance(result, Command) else result
                state = self._merge_state(state, updates)
            targets = []
            for task in completed:
                result = task["result"]
                if len(completed) == 1 and isinstance(result, Command) and result.goto:
                    routed = self._resolve_goto(result.goto)
                else:
                    routed = self._get_next_nodes(task["node"], state)
                for target in routed:
                    # Ordinary graph joins deduplicate names; Send slots never do.
                    if isinstance(target, Send) or target not in targets:
                        targets.append(target)
            tasks = self._plan_tasks(targets, state)
            after = next((task["node"] for task in completed
                          if task["node"] in self.interrupt_after), None)
            save({"interrupted_after": after} if after else None)
            if after:
                raise InterruptException({"node": after, "when": "after"})
            if not tasks:
                return state
        raise GraphRecursionError(f"Graph execution exceeded recursion limit of {recursion_limit} supersteps")

    async def _execute_node(
        self,
        func: Callable[..., Any],
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Command]:
        """Execute a single node function.
        
        Args:
            func: Node function to execute
            state: Current state
            config: Optional configuration (passed to nodes that accept it)
            
        Returns:
            State updates or Command
        """
        # Set cache context for task functions
        from upsonic.graphv2.task import _current_cache
        
        cache_token = None
        if self.cache:
            cache_token = _current_cache.set(self.cache)
        
        try:
            # Check function signature to see if it accepts config
            sig = inspect.signature(func)
            accepts_config = len(sig.parameters) >= 2
            
            # Check if function is async
            if inspect.iscoroutinefunction(func):
                if accepts_config and config is not None:
                    result = await func(state, config)
                else:
                    result = await func(state)
            else:
                if accepts_config and config is not None:
                    result = func(state, config)
                else:
                    result = func(state)
            
            return result
        finally:
            # Reset cache context
            if cache_token is not None:
                _current_cache.reset(cache_token)
    
    async def _execute_node_with_policies(
        self,
        node_name: str,
        node_config: NodeConfig,
        state: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Union[Dict[str, Any], Command, Send, List[Send]]:
        """Execute a node with retry and cache policies.
        
        Args:
            node_name: Name of the node
            node_config: Node configuration including policies
            state: Current state
            config: Runtime configuration
            
        Returns:
            Node result (updates, Command, or Send objects)
        """
        # Check cache if policy is set
        if node_config.cache_policy and self.cache:
            cache_key = node_config.cache_policy.key_func(state)
            cached_result = self.cache.get((node_name,), cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute with retry logic if policy is set
        if node_config.retry_policy and node_config.retry_safe:
            result = await self._execute_with_retry(
                node_config.func,
                state,
                node_config.retry_policy,
                config
            )
        else:
            result = await self._execute_node(node_config.func, state, config)
        
        # Store in cache if policy is set
        if node_config.cache_policy and self.cache:
            self.cache.put(
                (node_name,),
                cache_key,
                result,
                ttl=node_config.cache_policy.ttl
            )
        
        return result
    
    async def _execute_with_retry(
        self,
        func: Callable[..., Any],
        state: Dict[str, Any],
        retry_policy: RetryPolicy,
        config: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Command, Send, List[Send]]:
        """Execute a function with retry logic.
        
        Args:
            func: Function to execute
            state: Current state
            retry_policy: Retry policy to use
            config: Optional configuration
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        from upsonic.graphv2.task import calculate_retry_delay, should_retry
        
        last_exception = None
        
        for attempt in range(retry_policy.max_attempts):
            try:
                return await self._execute_node(func, state, config)
            except InterruptException:
                raise
            except Exception as e:
                last_exception = e
                
                # Check if we should retry
                if attempt < retry_policy.max_attempts - 1:
                    if should_retry(e, retry_policy):
                        delay = calculate_retry_delay(attempt, retry_policy)
                        await asyncio.sleep(delay)
                        continue
                
                # No retry or max attempts reached
                raise
        
        # This should not be reached
        if last_exception:
            raise last_exception
        raise ValueError("RetryPolicy.max_attempts must be greater than zero")
    
    def _resolve_goto(self, goto: Union[str, Send, List[Send], List[str]]) -> Union[List[str], List[Send]]:
        """Resolve goto target to list of node names or Send objects.
        
        When goto contains Send objects, they need special handling for dynamic parallelization.
        This method retains Send objects for the shared invocation planner.
        
        Args:
            goto: Goto target (node name, Send, or list of Send/node names)
            
        Returns:
            List of node names or Send objects to execute next
        """
        if isinstance(goto, str):
            return [goto]
        elif isinstance(goto, Send):
            # Return as a list containing the Send object
            # This will be detected and handled specially by the caller
            return [goto]
        elif isinstance(goto, list):
            # List can contain Send objects or strings
            # Return as-is, caller will handle appropriately
            return goto
        else:
            return [str(goto)]
    
    def _merge_state(
        self,
        current: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge state updates using reducers.
        
        Args:
            current: Current state
            updates: Updates to merge
            
        Returns:
            Merged state
        """
        merged = current.copy()
        
        for key, value in updates.items():
            if key in self.reducers:
                # Use reducer to merge
                reducer = self.reducers[key]
                if key in merged:
                    merged[key] = reducer(merged[key], value)
                else:
                    merged[key] = value
            else:
                # Simple replacement
                merged[key] = value
        
        return merged
    
    def _get_next_nodes(self, current_node: str, state: Dict[str, Any]) -> Union[List[str], List[Send]]:
        """Determine which nodes to execute next.
        
        Args:
            current_node: Name of node that just completed
            state: Current state
            
        Returns:
            List of node names to execute next, or list of Send objects for dynamic parallelization
        """
        # Check for conditional edge
        if current_node in self.conditional_routes:
            cond_edge = self.conditional_routes[current_node]
            result = cond_edge.condition(state)
            
            # Conditional edges can return:
            # - A string (node name)
            # - A Send object
            # - A list of Send objects
            if isinstance(result, list):
                # If it's a list, it could be a list of Send objects or strings
                return result
            elif isinstance(result, Send):
                # Single Send object - wrap in list
                return [result]
            else:
                # String or other - wrap in list
                return [result]
        
        # Check for normal edge(s)
        if current_node in self.normal_routes:
            return self.normal_routes[current_node]
        
        # No outgoing edges - end execution
        return [END]
    
    async def _resume_from_interrupt(
        self,
        command: Command,
        thread_id: str,
        checkpoint_id: str | None,
        config: dict[str, Any],
        recursion_limit: int
    ) -> dict[str, Any]:
        if not self.checkpointer:
            raise RuntimeError("Cannot resume without a checkpointer")
        checkpoint = self.checkpointer.get(thread_id, checkpoint_id)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        tasks = deepcopy(checkpoint.pending_tasks)
        if tasks is None:
            tasks = self._plan_tasks(checkpoint.next_nodes, checkpoint.state)
            for task in tasks:
                if not checkpoint.metadata.get("interrupted_before"):
                    task["status"] = "UNKNOWN"
        if checkpoint.metadata.get("interrupted_before"):
            approved_id = checkpoint.metadata.get("interrupted_task")
            if approved_id is None:
                # Legacy before-node checkpoints lacked a slot identity.
                approved_id = next((task["id"] for task in tasks
                                    if task["node"] == checkpoint.metadata["interrupted_before"]), None)
            for task in tasks:
                if task["id"] == approved_id:
                    task["before_approved"] = True
                    break
        interrupted = [task for task in tasks if task["status"] == "INTERRUPTED"]
        if interrupted:
            values = command.resume if isinstance(command.resume, list) else [command.resume]
            config = {**config, "_graph_resume_values": {
                task["id"]: values for task in tasks
                if task["status"] == "INTERRUPTED"}}
        return await self._execute_graph(
            deepcopy(checkpoint.state), checkpoint.next_nodes, thread_id,
            checkpoint.checkpoint_id, config, recursion_limit,
            pending_tasks=tasks)

    def get_state(self, config: Dict[str, Any]) -> Optional[StateSnapshot]:
        """Get the current state for a thread.
        
        Args:
            config: Configuration including thread_id and optional checkpoint_id
            
        Returns:
            StateSnapshot with current state, or None if not found
        """
        if not self.checkpointer:
            return None
        
        thread_config = config.get("configurable", {})
        thread_id = thread_config.get("thread_id")
        checkpoint_id = thread_config.get("checkpoint_id")
        
        if not thread_id:
            return None
        
        checkpoint = self.checkpointer.get(thread_id, checkpoint_id)
        if not checkpoint:
            return None
        
        return StateSnapshot(
            values=checkpoint.state,
            next=checkpoint.next_nodes,
            config={
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_id": checkpoint.checkpoint_id
                }
            },
            metadata=checkpoint.metadata,
            parent_config={
                "configurable": {
                    "thread_id": checkpoint.thread_id,
                    "checkpoint_id": checkpoint.parent_checkpoint_id
                }
            } if checkpoint.parent_checkpoint_id else None
        )
    
    def get_state_history(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None
    ) -> List[StateSnapshot]:
        """Get the state history for a thread.
        
        Args:
            config: Configuration including thread_id
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of StateSnapshots from newest to oldest
        """
        if not self.checkpointer:
            return []
        
        thread_config = config.get("configurable", {})
        thread_id = thread_config.get("thread_id")
        
        if not thread_id:
            return []
        
        checkpoints = self.checkpointer.get_history(thread_id, limit)
        
        return [
            StateSnapshot(
                values=checkpoint.state,
                next=checkpoint.next_nodes,
                config={
                    "configurable": {
                        "thread_id": checkpoint.thread_id,
                        "checkpoint_id": checkpoint.checkpoint_id
                    }
                },
                metadata=checkpoint.metadata,
                parent_config={
                    "configurable": {
                        "thread_id": checkpoint.thread_id,
                        "checkpoint_id": checkpoint.parent_checkpoint_id
                    }
                } if checkpoint.parent_checkpoint_id else None
            )
            for checkpoint in checkpoints
        ]
    
    def update_state(
        self,
        config: Dict[str, Any],
        values: Dict[str, Any],
        as_node: Optional[str] = None
    ) -> None:
        """Update the state at a checkpoint.
        
        Args:
            config: Configuration including thread_id and optional checkpoint_id
            values: State updates to apply
            as_node: Optional node name that "wrote" these updates
        """
        if not self.checkpointer:
            raise RuntimeError("Cannot update state without a checkpointer")
        
        thread_config = config.get("configurable", {})
        thread_id = thread_config.get("thread_id")
        checkpoint_id = thread_config.get("checkpoint_id")
        
        if not thread_id:
            raise ValueError("thread_id required in config")
        
        # Get the current checkpoint
        checkpoint = self.checkpointer.get(thread_id, checkpoint_id)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        
        latest = self.checkpointer.get(thread_id)
        if latest and latest.pending_tasks:
            raise GraphExecutionUncertainError(latest.pending_tasks)
        if checkpoint.pending_tasks:
            raise GraphExecutionUncertainError(checkpoint.pending_tasks)

        # Merge the updates
        updated_state = self._merge_state(checkpoint.state, values)
        
        # Create a new checkpoint with the updated state
        new_checkpoint = Checkpoint(
            checkpoint_id=generate_checkpoint_id(),
            thread_id=thread_id,
            state=updated_state,
            next_nodes=checkpoint.next_nodes,
            parent_checkpoint_id=checkpoint.checkpoint_id,
            metadata={**checkpoint.metadata, **({"updated_by": as_node} if as_node else {})},
            pending_tasks=deepcopy(checkpoint.pending_tasks)
        )
        
        self.checkpointer.put(new_checkpoint)
    
    def get_graph(self) -> "RunnableGraph":
        """Get a graph representation of this compiled state graph for visualization.
        
        Returns a RunnableGraph that can render the graph structure as
        ASCII art or Mermaid diagrams. Normal edges are shown as solid
        connections, conditional edges as dashed connections.
        
        Returns:
            A RunnableGraph object with to_ascii(), to_mermaid(), and
            get_structure_details() methods
        """
        from upsonic.uel.graph import RunnableGraph
        return RunnableGraph(self)

