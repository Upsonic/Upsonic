"""
StateGraph - The core graph execution engine.

This module provides the StateGraph class which enables building stateful,
multi-step workflows with LLMs using a graph-based approach.
"""

from __future__ import annotations

import asyncio
import inspect
import operator
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

from upsonic.uel.runnable import Runnable
from upsonic.graphv2.checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    StateSnapshot,
    generate_checkpoint_id,
)
from upsonic.graphv2.primitives import Command, END, InterruptException, Send
from upsonic.graphv2.store import BaseStore
from upsonic.graphv2.cache import BaseCache, CachePolicy
from upsonic.graphv2.task import RetryPolicy
from upsonic.graphv2.errors import GraphRecursionError, GraphValidationError


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
    
    func: Callable[[Dict[str, Any]], Any]
    retry_policy: Optional[RetryPolicy] = None
    cache_policy: Optional[CachePolicy] = None


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
        func: Callable[[Dict[str, Any]], Any],
        *,
        retry_policy: Optional[RetryPolicy] = None,
        cache_policy: Optional[CachePolicy] = None
    ) -> None:
        """Add a node to the graph.
        
        Args:
            name: Unique name for the node
            func: Function that takes state and returns state updates
            retry_policy: Optional retry configuration
            cache_policy: Optional cache configuration
            
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
            cache_policy=cache_policy
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
        """Execute the graph asynchronously.
        
        Args:
            input: Initial state or Command to resume from interrupt (None to resume from failure)
            config: Configuration including thread_id, checkpoint_id, and recursion_limit
            context: Runtime context parameters
            
        Returns:
            Final state dictionary
        """
        # Get or create thread configuration
        config = config or {}
        thread_config = config.get("configurable", {})
        thread_id = thread_config.get("thread_id", "default")
        checkpoint_id = thread_config.get("checkpoint_id")
        recursion_limit = config.get("recursion_limit", 100)
        
        # Store context in config for node access
        if context:
            config["context"] = context
        
        # Validate input against input_schema if provided
        if input is not None and not isinstance(input, Command) and self.input_schema:
            input = self._validate_input(input)
        
        # Handle resume from interrupt
        if isinstance(input, Command):
            try:
                final_state = await self._resume_from_interrupt(input, thread_id, checkpoint_id, config, recursion_limit)
                
                # Flush pending checkpoints if using exit durability
                if self.durability == "exit" and self.checkpointer:
                    await self._flush_exit_checkpoints()
                
                return self._filter_output(final_state)
            except InterruptException as e:
                # A subsequent node interrupted during resume
                # Flush pending checkpoints if using exit durability
                if self.durability == "exit" and self.checkpointer:
                    await self._flush_exit_checkpoints()
                
                # Get the current state from the latest checkpoint
                state = {}
                if self.checkpointer:
                    latest_checkpoint = self.checkpointer.get(thread_id)
                    if latest_checkpoint:
                        state = latest_checkpoint.state
                return {
                    **state,
                    "__interrupt__": [{"value": e.value}]
                }
        
        # Handle resume from failure (input is None)
        if input is None and self.checkpointer:
            existing_checkpoint = self.checkpointer.get(thread_id, checkpoint_id)
            if existing_checkpoint:
                # Resume from where we left off
                state = deepcopy(existing_checkpoint.state)
                next_nodes = existing_checkpoint.next_nodes
                parent_checkpoint_id = existing_checkpoint.checkpoint_id
            else:
                raise ValueError(f"Cannot resume: no checkpoint found for thread_id={thread_id}")
        # Load existing state from checkpoint or start fresh
        elif self.checkpointer and (existing_checkpoint := self.checkpointer.get(thread_id, checkpoint_id)):
            state = deepcopy(existing_checkpoint.state)
            next_nodes = existing_checkpoint.next_nodes
            parent_checkpoint_id = existing_checkpoint.checkpoint_id
            
            # If input provided, merge it and restart from START
            if input:
                state = self._merge_state(state, input)
                # If the previous execution ended (next_nodes is END), restart from START
                if END in next_nodes or not next_nodes:
                    next_nodes = [START]
        else:
            # Fresh start
            state = input or {}
            next_nodes = [START]
            parent_checkpoint_id = None
        
        # Execute the graph
        try:
            final_state = await self._execute_graph(
                state=state,
                next_nodes=next_nodes,
                thread_id=thread_id,
                parent_checkpoint_id=parent_checkpoint_id,
                config=config,
                recursion_limit=recursion_limit
            )
            
            # Flush pending checkpoints if using exit durability
            if self.durability == "exit" and self.checkpointer:
                await self._flush_exit_checkpoints()
            
            return self._filter_output(final_state)
            
        except InterruptException as e:
            # Graph was interrupted - the checkpoint is already saved
            # Flush pending checkpoints if using exit durability
            if self.durability == "exit" and self.checkpointer:
                await self._flush_exit_checkpoints()
            
            # Just return the current state with interrupt marker
            return {
                **state,
                "__interrupt__": [{"value": e.value}]
            }
        except Exception as e:
            # Any other error - flush checkpoints if using exit durability
            if self.durability == "exit" and self.checkpointer:
                await self._flush_exit_checkpoints()
            # Re-raise the exception
            raise
    
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
    
    async def _execute_graph(
        self,
        state: Dict[str, Any],
        next_nodes: List[str],
        thread_id: str,
        parent_checkpoint_id: Optional[str],
        config: Dict[str, Any],
        recursion_limit: int
    ) -> Dict[str, Any]:
        """Execute the graph from the given state and nodes.
        
        Args:
            state: Current state
            next_nodes: List of nodes to execute next
            thread_id: Thread identifier
            parent_checkpoint_id: Parent checkpoint for history
            config: Configuration including context
            recursion_limit: Maximum number of supersteps allowed
            
        Returns:
            Final state after execution completes
            
        Raises:
            GraphRecursionError: If recursion_limit is exceeded
        """
        current_nodes = next_nodes
        current_parent = parent_checkpoint_id
        superstep_count = 0
        
        while current_nodes:
            # Check recursion limit
            superstep_count += 1
            if superstep_count > recursion_limit:
                raise GraphRecursionError(
                    f"Graph execution exceeded recursion limit of {recursion_limit} supersteps"
                )
            # Get the first node to execute
            if START in current_nodes:
                # Start node - find what comes after it
                next_result = self._get_next_nodes(START, state)
                # Check if we got Send objects
                if next_result and len(next_result) > 0 and isinstance(next_result[0], Send):
                    # Execute Send objects from START
                    send_results = await self._execute_send_objects(next_result, state, config)
                    for send_updates in send_results:
                        state = self._merge_state(state, send_updates)
                    # After Send execution, determine next nodes
                    all_next = set()
                    for send_obj in next_result:
                        next_from_send = self._get_next_nodes(send_obj.node, state)
                        if next_from_send and not isinstance(next_from_send[0], Send):
                            all_next.update(next_from_send)
                    current_nodes = list(all_next) if all_next else [END]
                else:
                    current_nodes = next_result
                continue
            
            if END in current_nodes or not current_nodes:
                # Reached the end
                break
            
            # Execute all ready nodes in parallel if multiple
            nodes_to_execute = [n for n in current_nodes if n != END]
            
            if not nodes_to_execute:
                break
            
            if len(nodes_to_execute) == 1:
                # Single node execution
                node_name = nodes_to_execute[0]
                
                # Check for interrupt before
                if node_name in self.interrupt_before:
                    await self._save_checkpoint(
                        state=state,
                        next_nodes=current_nodes,
                        thread_id=thread_id,
                        parent_checkpoint_id=current_parent,
                        metadata={"interrupted_before": node_name}
                    )
                    raise InterruptException({"node": node_name, "when": "before"})
                
                # Execute the node with retry and caching
                node_config = self.nodes[node_name]
                try:
                    updates = await self._execute_node_with_policies(
                        node_name=node_name,
                        node_config=node_config,
                        state=state,
                        config=config
                    )
                except InterruptException as e:
                    # Node called interrupt() - save checkpoint and re-raise
                    await self._save_checkpoint(
                        state=state,
                        next_nodes=[node_name],  # Will re-execute this node on resume
                        thread_id=thread_id,
                        parent_checkpoint_id=current_parent,
                        metadata={"interrupted_node": node_name, "interrupt_value": e.value}
                    )
                    raise
                except Exception as e:
                    # Node failed with an error - save checkpoint for recovery
                    if self.checkpointer:
                        await self._save_checkpoint(
                            state=state,
                            next_nodes=[node_name],  # Will re-execute this node on resume
                            thread_id=thread_id,
                            parent_checkpoint_id=current_parent,
                            metadata={"failed_node": node_name, "error": str(e)}
                        )
                    raise
                
                # Check if node returned a Command
                next_goto = None
                
                if isinstance(updates, Command):
                    next_goto = updates.goto
                    updates = updates.update or {}
                elif isinstance(updates, (Send, list)):
                    # Node returned Send objects directly - this is NOT supported!
                    # Send objects should ONLY be returned from conditional edge functions
                    if isinstance(updates, Send) or (isinstance(updates, list) and any(isinstance(u, Send) for u in updates)):
                        raise ValueError(
                            f"Node '{node_name}' returned Send object(s) directly. "
                            "Send objects can only be returned from conditional edge functions. "
                            "Use add_conditional_edges() with a routing function that returns Send objects."
                        )
                
                # Merge updates into state
                state = self._merge_state(state, updates)
                
                # Check for interrupt after
                if node_name in self.interrupt_after:
                    # Determine next nodes before interrupting
                    if next_goto:
                        resolved_goto = self._resolve_goto(next_goto)
                        # Convert Send objects to node names for checkpoint
                        if resolved_goto and isinstance(resolved_goto[0], Send):
                            next_nodes_after = [s.node for s in resolved_goto if isinstance(s, Send)]
                        else:
                            next_nodes_after = resolved_goto
                    else:
                        next_nodes_result = self._get_next_nodes(node_name, state)
                        # Convert Send objects to node names for checkpoint
                        if next_nodes_result and isinstance(next_nodes_result[0], Send):
                            next_nodes_after = [s.node for s in next_nodes_result if isinstance(s, Send)]
                        else:
                            next_nodes_after = next_nodes_result
                    
                    await self._save_checkpoint(
                        state=state,
                        next_nodes=next_nodes_after,
                        thread_id=thread_id,
                        parent_checkpoint_id=current_parent,
                        metadata={"interrupted_after": node_name}
                    )
                    raise InterruptException({"node": node_name, "when": "after"})
                
                # Save checkpoint after super-step based on durability mode
                if self.checkpointer:
                    if next_goto:
                        resolved_goto = self._resolve_goto(next_goto)
                        # Convert Send objects to node names for checkpoint
                        if resolved_goto and isinstance(resolved_goto[0], Send):
                            next_nodes_to_save = [s.node for s in resolved_goto if isinstance(s, Send)]
                        else:
                            next_nodes_to_save = resolved_goto
                    else:
                        next_nodes_result = self._get_next_nodes(node_name, state)
                        # Convert Send objects to node names for checkpoint
                        if next_nodes_result and isinstance(next_nodes_result[0], Send):
                            next_nodes_to_save = [s.node for s in next_nodes_result if isinstance(s, Send)]
                        else:
                            next_nodes_to_save = next_nodes_result
                    
                    checkpoint = await self._save_checkpoint_with_durability(
                        state=state,
                        next_nodes=next_nodes_to_save,
                        thread_id=thread_id,
                        parent_checkpoint_id=current_parent
                    )
                    current_parent = checkpoint.checkpoint_id
                
                # Determine next nodes
                if next_goto:
                    resolved_goto = self._resolve_goto(next_goto)
                    # Check if we have Send objects to execute
                    if resolved_goto and isinstance(resolved_goto[0], Send):
                        # Execute Send objects and continue with remaining edges
                        send_results = await self._execute_send_objects(
                            send_objects=resolved_goto,
                            state=state,
                            config=config
                        )
                        # Merge send results
                        for send_updates in send_results:
                            state = self._merge_state(state, send_updates)
                        # After Send objects execute, get next nodes from Send targets
                        all_next = set()
                        for send_obj in resolved_goto:
                            next_from_send = self._get_next_nodes(send_obj.node, state)
                            # Make sure we don't get more Send objects (only one level)
                            if next_from_send and not isinstance(next_from_send[0], Send):
                                all_next.update(next_from_send)
                        current_nodes = list(all_next) if all_next else [END]
                    else:
                        current_nodes = resolved_goto
                else:
                    next_result = self._get_next_nodes(node_name, state)
                    # Check if we got Send objects from conditional edge
                    if next_result and len(next_result) > 0 and isinstance(next_result[0], Send):
                        # Execute Send objects
                        send_results = await self._execute_send_objects(next_result, state, config)
                        for send_updates in send_results:
                            state = self._merge_state(state, send_updates)
                        # After Send execution, determine actual next nodes
                        # Get next nodes from all Send target nodes
                        all_next = set()
                        for send_obj in next_result:
                            next_from_send = self._get_next_nodes(send_obj.node, state)
                            # Make sure we don't get more Send objects (only one level)
                            if next_from_send and not isinstance(next_from_send[0], Send):
                                all_next.update(next_from_send)
                        current_nodes = list(all_next) if all_next else [END]
                    else:
                        current_nodes = next_result
            
            else:
                # Multiple nodes - execute in parallel
                parallel_results = await self._execute_nodes_parallel(
                    nodes_to_execute, state, thread_id, current_parent, config
                )
                
                # Merge all updates from parallel execution
                for node_name, updates in parallel_results:
                    if isinstance(updates, Command):
                        # Command in parallel execution - apply updates only
                        # (routing is handled after all parallel nodes complete)
                        if updates.update:
                            state = self._merge_state(state, updates.update)
                    else:
                        state = self._merge_state(state, updates)
                
                # Save checkpoint after parallel super-step
                if self.checkpointer:
                    # Determine next nodes from all completed parallel nodes
                    all_next_nodes = set()
                    for node_name, _ in parallel_results:
                        next_from_node = self._get_next_nodes(node_name, state)
                        # Convert Send objects to node names for checkpoint
                        if next_from_node and isinstance(next_from_node[0], Send):
                            all_next_nodes.update(s.node for s in next_from_node if isinstance(s, Send))
                        else:
                            all_next_nodes.update(next_from_node)
                    
                    next_nodes_to_save = list(all_next_nodes)
                    
                    checkpoint = await self._save_checkpoint(
                        state=state,
                        next_nodes=next_nodes_to_save,
                        thread_id=thread_id,
                        parent_checkpoint_id=current_parent,
                        metadata={"parallel_execution": nodes_to_execute}
                    )
                    current_parent = checkpoint.checkpoint_id
                
                # Determine next nodes from all completed nodes
                all_next_nodes = set()
                send_objects_to_execute = []
                
                for node_name, _ in parallel_results:
                    next_from_node = self._get_next_nodes(node_name, state)
                    # Check if we got Send objects
                    if next_from_node and len(next_from_node) > 0 and isinstance(next_from_node[0], Send):
                        # Collect Send objects to execute later
                        send_objects_to_execute.extend(next_from_node)
                    else:
                        all_next_nodes.update(next_from_node)
                
                # Execute any Send objects from conditional edges after parallel nodes
                if send_objects_to_execute:
                    send_results = await self._execute_send_objects(send_objects_to_execute, state, config)
                    for send_updates in send_results:
                        state = self._merge_state(state, send_updates)
                    # Get next nodes from Send targets
                    for send_obj in send_objects_to_execute:
                        next_from_send = self._get_next_nodes(send_obj.node, state)
                        if next_from_send and not isinstance(next_from_send[0], Send):
                            all_next_nodes.update(next_from_send)
                
                current_nodes = list(all_next_nodes) if all_next_nodes else [END]
        
        return state
    
    async def _execute_nodes_parallel(
        self,
        node_names: List[str],
        state: Dict[str, Any],
        thread_id: str,
        parent_checkpoint_id: Optional[str],
        config: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, Union[Dict[str, Any], Command]]]:
        """Execute multiple nodes in parallel.
        
        Args:
            node_names: List of node names to execute
            state: Current state (read-only for all nodes)
            thread_id: Thread identifier
            parent_checkpoint_id: Parent checkpoint for history
            config: Optional configuration
            
        Returns:
            List of (node_name, updates) tuples
        """
        # Create tasks for all nodes
        tasks = []
        for node_name in node_names:
            node_config = self.nodes[node_name]
            # Use _execute_node_with_policies to support retry and cache
            task = self._execute_node_with_policies(node_name, node_config, state, config or {})
            tasks.append((node_name, task))
        
        # Execute all tasks truly in parallel using asyncio.gather
        # We need to use return_exceptions=False so exceptions are raised
        task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Process results and handle errors
        results = []
        for i, (node_name, _) in enumerate(tasks):
            result = task_results[i]
            
            if isinstance(result, InterruptException):
                # Interrupts not supported in parallel execution
                raise RuntimeError(
                    f"Node '{node_name}' called interrupt() during parallel execution. "
                    "Interrupts are not supported in parallel execution."
                )
            elif isinstance(result, Exception):
                # Handle node errors - re-raise with context
                raise RuntimeError(f"Error in parallel node '{node_name}': {str(result)}") from result
            else:
                results.append((node_name, result))
        
        return results
    
    async def _execute_node(
        self,
        func: Callable[[Dict[str, Any]], Any],
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
        if node_config.retry_policy:
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
        func: Callable[[Dict[str, Any]], Any],
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
        from upsonic.graphv2.task import should_retry, calculate_retry_delay
        
        last_exception = None
        
        for attempt in range(retry_policy.max_attempts):
            try:
                return await self._execute_node(func, state, config)
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
    
    async def _execute_send_objects(
        self,
        send_objects: List[Send],
        state: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute Send objects (dynamic parallelization).
        
        Args:
            send_objects: List of Send objects to execute
            state: Current state (base state for all Send tasks)
            config: Runtime configuration
            
        Returns:
            List of state updates from all Send tasks
        """
        # Create tasks for all Send objects
        tasks = []
        send_metadata = []  # Store metadata for error messages
        for send_obj in send_objects:
            # Get the node to invoke
            if send_obj.node not in self.nodes:
                raise ValueError(f"Send references unknown node: {send_obj.node}")
            
            node_config = self.nodes[send_obj.node]
            
            # Merge Send state with base state
            send_state = {**state, **send_obj.state}
            
            # Execute the node
            task = self._execute_node_with_policies(
                node_name=send_obj.node,
                node_config=node_config,
                state=send_state,
                config=config
            )
            tasks.append(task)
            send_metadata.append(send_obj.node)
        
        # Execute all tasks truly in parallel using asyncio.gather
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle errors
        results = []
        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                # Handle errors in Send tasks
                node_name = send_metadata[i]
                raise RuntimeError(f"Error in Send task for node '{node_name}': {str(result)}") from result
            else:
                # Extract updates from result
                if isinstance(result, Command):
                    updates = result.update or {}
                elif isinstance(result, dict):
                    updates = result
                else:
                    updates = {}
                results.append(updates)
        
        return results
    
    def _resolve_goto(self, goto: Union[str, Send, List[Send], List[str]]) -> Union[List[str], List[Send]]:
        """Resolve goto target to list of node names or Send objects.
        
        When goto contains Send objects, they need special handling for dynamic parallelization.
        This method returns Send objects as-is so they can be processed by _execute_send_objects.
        
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
    
    async def _save_checkpoint_with_durability(
        self,
        state: Dict[str, Any],
        next_nodes: List[str],
        thread_id: str,
        parent_checkpoint_id: Optional[str]
    ) -> Checkpoint:
        """Save checkpoint based on durability mode.
        
        Args:
            state: Current state
            next_nodes: Nodes to execute next
            thread_id: Thread identifier
            parent_checkpoint_id: Parent checkpoint ID
            
        Returns:
            The saved checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=generate_checkpoint_id(),
            thread_id=thread_id,
            state=deepcopy(state),
            next_nodes=next_nodes,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata={}
        )
        
        if self.durability == "sync":
            # Save synchronously - wait for completion before continuing
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.checkpointer.put,
                checkpoint
            )
        elif self.durability == "async":
            # Save asynchronously - schedule in background and continue immediately
            # The checkpoint will be saved while the next node executes
            asyncio.get_event_loop().run_in_executor(
                None,
                self.checkpointer.put,
                checkpoint
            )
        elif self.durability == "exit":
            # Don't save during execution - only on exit
            # Store for later saving
            if not hasattr(self, '_pending_checkpoints'):
                self._pending_checkpoints = []
            self._pending_checkpoints.append(checkpoint)
        
        return checkpoint
    
    async def _flush_exit_checkpoints(self):
        """Flush pending checkpoints when using exit durability mode.
        
        This is called when the graph completes or encounters an error
        in exit durability mode.
        """
        if hasattr(self, '_pending_checkpoints') and self._pending_checkpoints:
            # Save all pending checkpoints
            for checkpoint in self._pending_checkpoints:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.checkpointer.put,
                    checkpoint
                )
            self._pending_checkpoints = []
    
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
    
    async def _save_checkpoint(
        self,
        state: Dict[str, Any],
        next_nodes: List[str],
        thread_id: str,
        parent_checkpoint_id: Optional[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """Save a checkpoint.
        
        Args:
            state: Current state
            next_nodes: Nodes to execute next
            thread_id: Thread identifier
            parent_checkpoint_id: Parent checkpoint ID
            metadata: Optional metadata about the checkpoint
            
        Returns:
            The saved checkpoint
        """
        checkpoint = Checkpoint(
            checkpoint_id=generate_checkpoint_id(),
            thread_id=thread_id,
            state=deepcopy(state),
            next_nodes=next_nodes,
            parent_checkpoint_id=parent_checkpoint_id,
            metadata=metadata or {}
        )
        
        if self.checkpointer:
            self.checkpointer.put(checkpoint)
        
        return checkpoint
    
    async def _resume_from_interrupt(
        self,
        command: Command,
        thread_id: str,
        checkpoint_id: Optional[str],
        config: Dict[str, Any],
        recursion_limit: int
    ) -> Dict[str, Any]:
        """Resume execution from an interrupt.
        
        This method properly handles resuming from an interrupt by:
        1. Loading the checkpoint where execution was interrupted
        2. Setting up the context so interrupt() calls return resume values
        3. Re-executing the interrupted node(s)
        4. Continuing normal execution
        
        Args:
            command: Command with resume value
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint to resume from
            
        Returns:
            Final state after resuming
        """
        from upsonic.graphv2.primitives import _interrupt_resume_values, _interrupt_counter
        
        if not self.checkpointer:
            raise RuntimeError("Cannot resume without a checkpointer")
        
        # Load the checkpoint
        checkpoint = self.checkpointer.get(thread_id, checkpoint_id)
        if not checkpoint:
            raise ValueError(f"No checkpoint found for thread_id={thread_id}")
        
        state = deepcopy(checkpoint.state)
        
        # Check if this checkpoint has interrupt metadata
        interrupted_before = checkpoint.metadata.get("interrupted_before")
        interrupted_after = checkpoint.metadata.get("interrupted_after")
        
        if interrupted_before:
            # The node was interrupted before execution
            # Just continue normally - the node hasn't run yet
            final_state = await self._execute_graph(
                state=state,
                next_nodes=checkpoint.next_nodes,
                thread_id=thread_id,
                parent_checkpoint_id=checkpoint.checkpoint_id,
                config=config,
                recursion_limit=recursion_limit
            )
            return final_state
        
        elif interrupted_after:
            # The node was interrupted after execution  
            # Just continue from next_nodes
            final_state = await self._execute_graph(
                state=state,
                next_nodes=checkpoint.next_nodes,
                thread_id=thread_id,
                parent_checkpoint_id=checkpoint.checkpoint_id,
                config=config,
                recursion_limit=recursion_limit
            )
            return final_state
        
        else:
            # This was an interrupt from inside a node function
            # We need to re-execute the node with resume context
            
            # Determine which node to re-execute
            # The checkpoint's next_nodes should contain the node that was executing
            if not checkpoint.next_nodes:
                raise RuntimeError("Cannot determine interrupted node from checkpoint")
            
            interrupted_node = checkpoint.next_nodes[0]
            
            # Set up the interrupt resume context
            # Convert resume value to list if it isn't already
            if isinstance(command.resume, list):
                resume_values = command.resume
            else:
                resume_values = [command.resume]
            
            # Set the context variables for this execution
            resume_token = _interrupt_resume_values.set(resume_values)
            counter_token = _interrupt_counter.set(0)
            
            try:
                # Re-execute the interrupted node with resume context
                node_config = self.nodes[interrupted_node]
                updates = await self._execute_node(node_config.func, state)
                
                # Process the results
                next_goto = None
                if isinstance(updates, Command):
                    next_goto = updates.goto
                    updates = updates.update or {}
                
                # Merge updates
                state = self._merge_state(state, updates)
                
                # Determine next nodes
                if next_goto:
                    next_nodes = [next_goto] if next_goto != END else [END]
                else:
                    next_nodes = self._get_next_nodes(interrupted_node, state)
                
                # Save checkpoint after completing the interrupted node
                if self.checkpointer:
                    new_checkpoint = await self._save_checkpoint(
                        state=state,
                        next_nodes=next_nodes,
                        thread_id=thread_id,
                        parent_checkpoint_id=checkpoint.checkpoint_id,
                        metadata={"resumed_from": checkpoint.checkpoint_id}
                    )
                    parent_checkpoint_id = new_checkpoint.checkpoint_id
                else:
                    parent_checkpoint_id = checkpoint.checkpoint_id
                
            finally:
                # Clean up the resume context before continuing
                # This ensures subsequent nodes start fresh
                _interrupt_resume_values.reset(resume_token)
                _interrupt_counter.reset(counter_token)
            
            # Continue execution from the next nodes WITHOUT resume context
            # Any new interrupts in subsequent nodes will be fresh interrupts
            final_state = await self._execute_graph(
                state=state,
                next_nodes=next_nodes,
                thread_id=thread_id,
                parent_checkpoint_id=parent_checkpoint_id,
                config=config,
                recursion_limit=recursion_limit
            )
            
            return final_state
    
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
        
        # Merge the updates
        updated_state = self._merge_state(checkpoint.state, values)
        
        # Create a new checkpoint with the updated state
        new_checkpoint = Checkpoint(
            checkpoint_id=generate_checkpoint_id(),
            thread_id=thread_id,
            state=updated_state,
            next_nodes=checkpoint.next_nodes,
            parent_checkpoint_id=checkpoint.checkpoint_id,
            metadata={"updated_by": as_node} if as_node else {}
        )
        
        self.checkpointer.put(new_checkpoint)

