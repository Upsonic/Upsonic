from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Union
import copy

from pydantic import BaseModel, Field, ConfigDict
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn)
from rich.table import Table

from ..agent.base import BaseAgent
try:
    from ..storage.base import Storage
except ImportError:
    Storage = None
from ..tasks.tasks import Task
from ..utils.printing import console, escape_rich_markup, spacing

# This import is essential for the Graph's topological duty.
from ..context.sources import TaskOutputSource


class DecisionResponse(BaseModel):
    """Response type for LLM-based decisions that returns a boolean result."""
    result: bool


class DecisionLLM(BaseModel):
    """
    A decision node that uses a language model to evaluate input and determine execution flow.

    Attributes:
        description: Human-readable description of the decision
        true_branch: The branch to follow if the LLM decides yes/true
        false_branch: The branch to follow if the LLM decides no/false
        id: Unique identifier for this decision node
    """
    description: str
    true_branch: Optional[Union['TaskNode', 'TaskChain', 'DecisionFunc', 'DecisionLLM']] = None
    false_branch: Optional[Union['TaskNode', 'TaskChain', 'DecisionFunc', 'DecisionLLM']] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, description: str, *, true_branch=None, false_branch=None, id=None, **kwargs):
        """
        Initialize a DecisionLLM with a positional description parameter.
        
        Args:
            description: Human-readable description of the decision
            true_branch: The branch to follow if the LLM decides yes/true
            false_branch: The branch to follow if the LLM decides no/false
            id: Unique identifier for this decision node
        """
        if id is None: id = str(uuid.uuid4())
        super().__init__(description=description, true_branch=true_branch, false_branch=false_branch, id=id, **kwargs)
    
    async def evaluate(self, data: Any) -> bool:
        """
        Evaluates the decision using an LLM with the provided data.
        
        This is a placeholder that will be replaced during graph execution with
        actual LLM inference using the graph's default agent.
        
        Args:
            data: Data to evaluate (typically the output of the previous task)
            
        Returns:
            True if the LLM determines yes/true, False otherwise
        """
        # Placeholder; actual implementation is in Graph._evaluate_decision
        return True
    
    def _generate_prompt(self, data: Any) -> str:
        """
        Generates a prompt for the LLM based on the decision description and input data.
        
        Args:
            data: The data to be evaluated (typically the output of the previous task)
            
        Returns:
            A formatted prompt string for the LLM
        """
        prompt = f"""
You are an decision node in a graph.

Decision question: {self.description}

Previous node output:
<data>
{data}
</data>
"""
        return prompt.strip()
    
    def if_true(self, branch: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'DecisionLLM':
        """
        Sets the branch to follow if the LLM evaluates to True/Yes.
        
        Args:
            branch: The node, task, or chain to execute if the LLM decides yes/true
            
        Returns:
            Self for method chaining
        """
        if isinstance(branch, Task): branch = TaskNode(task=branch)
        self.true_branch = branch
        return self
    
    def if_false(self, branch: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'DecisionLLM':
        """
        Sets the branch to follow if the LLM evaluates to False/No.
        
        Args:
            branch: The node, task, or chain to execute if the LLM decides no/false
            
        Returns:
            Self for method chaining
        """
        if isinstance(branch, Task): branch = TaskNode(task=branch)
        self.false_branch = branch
        return self
    
    def __rshift__(self, other: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'TaskChain':
        """
        Implements the >> operator to chain this decision with another node,
        creating a new TaskChain.
        
        Args:
            other: The node, task, or chain to connect after this decision.
            
        Returns:
            A new TaskChain object representing the connection.
        """
        chain = TaskChain()
        chain.add(self)
        chain.add(other)
        return chain


class DecisionFunc(BaseModel):
    """
    A decision node that evaluates a condition function on task output to determine execution flow.

    Attributes:
        description: Human-readable description of the decision
        func: The function that evaluates the condition
        true_branch: The branch to follow if the condition is true
        false_branch: The branch to follow if the condition is false
        id: Unique identifier for this decision node
    """
    description: str
    func: Callable
    true_branch: Optional[Union['TaskNode', 'TaskChain', 'DecisionFunc', 'DecisionLLM']] = None
    false_branch: Optional[Union['TaskNode', 'TaskChain', 'DecisionFunc', 'DecisionLLM']] = None
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def __init__(self, description: str, func: Callable, *, true_branch=None, false_branch=None, id=None, **kwargs):
        """
        Initialize a DecisionFunc with positional description and func parameters.
        
        Args:
            description: Human-readable description of the decision
            func: The function that evaluates the condition
            true_branch: The branch to follow if the condition is true
            false_branch: The branch to follow if the condition is false
            id: Unique identifier for this decision node
        """
        if id is None: id = str(uuid.uuid4())
        super().__init__(description=description, func=func, true_branch=true_branch, false_branch=false_branch, id=id, **kwargs)
        
    def evaluate(self, data: Any) -> bool:
        """
        Evaluates the condition function with the provided data.
        
        Args:
            data: Data to evaluate (typically the output of the previous task)
            
        Returns:
            True if condition passes, False otherwise
        """
        return self.func(data)
    
    def if_true(self, branch: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'DecisionFunc':
        """
        Sets the branch to follow if the condition evaluates to True.
        
        Args:
            branch: The node, task, or chain to execute if condition is true
            
        Returns:
            Self for method chaining
        """
        if isinstance(branch, Task): branch = TaskNode(task=branch)
        self.true_branch = branch
        return self
    
    def if_false(self, branch: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'DecisionFunc':
        """
        Sets the branch to follow if the condition evaluates to False.
        
        Args:
            branch: The node, task, or chain to execute if condition is false
            
        Returns:
            Self for method chaining
        """
        if isinstance(branch, Task): branch = TaskNode(task=branch)
        self.false_branch = branch
        return self
    
    def __rshift__(self, other: Union['TaskNode', Task, 'TaskChain', 'DecisionFunc', 'DecisionLLM']) -> 'TaskChain':
        """
        Implements the >> operator to chain this decision with another node,
        creating a new TaskChain.
        
        Args:
            other: The node, task, or chain to connect after this decision.
            
        Returns:
            A new TaskChain object representing the connection.
        """
        chain = TaskChain()
        chain.add(self)
        chain.add(other)
        return chain


class TaskNode(BaseModel):
    """
    Wrapper around a Task that adds graph connectivity features.
    
    Attributes:
        task: The Task object this node wraps
        id: Unique identifier for this node
    """
    task: Task
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    def __rshift__(self, other: Union['TaskNode', Task, 'DecisionFunc', 'DecisionLLM', 'TaskChain']) -> 'TaskChain':
        """
        Implements the >> operator to connect nodes in a chain.
        
        Args:
            other: The next node, task, or chain in the chain
            
        Returns:
            A TaskChain object containing both nodes
        """
        chain = TaskChain()
        chain.add(self)
        chain.add(other)
        return chain


class TaskChain:
    """
    Represents a chain of connected task nodes.
    
    Attributes:
        nodes: List of nodes in the chain
        edges: Dictionary mapping node IDs to their next nodes. This is the single source of truth for graph topology.
    """
    def __init__(self):
        self.nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = []
        self.edges: Dict[str, List[str]] = {}
        
    def _get_leaf_nodes(self) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """Finds all nodes in the chain that have no outgoing edges."""
        if not self.nodes: return []
        source_node_ids = set(self.edges.keys())
        return [node for node in self.nodes if node.id not in source_node_ids]

    def add(self, node_or_chain: Union[TaskNode, Task, 'TaskChain', DecisionFunc, DecisionLLM]) -> 'TaskChain':
        """
        Adds a node or another chain to this chain, connecting it to the current leaf nodes.
        This method correctly handles branching and convergence.
        
        Args:
            node_or_chain: The node, task, or chain to add.
            
        Returns:
            This chain for method chaining.
        """
        previous_leaves = self._get_leaf_nodes()
        if isinstance(node_or_chain, Task): node_or_chain = TaskNode(task=node_or_chain)

        entry_points = []
        if isinstance(node_or_chain, (TaskNode, DecisionFunc, DecisionLLM)):
            new_node = node_or_chain
            if new_node not in self.nodes: self.nodes.append(new_node)
            entry_points.append(new_node)
            if isinstance(new_node, (DecisionFunc, DecisionLLM)):
                for branch in [new_node.true_branch, new_node.false_branch]:
                    if not branch: continue
                    if isinstance(branch, TaskChain):
                        self.nodes.extend(n for n in branch.nodes if n not in self.nodes)
                        self.edges.update(branch.edges)
                        if branch.nodes:
                            if new_node.id not in self.edges: self.edges[new_node.id] = []
                            self.edges[new_node.id].append(branch.nodes[0].id)
                    else:
                        if branch not in self.nodes: self.nodes.append(branch)
                        if new_node.id not in self.edges: self.edges[new_node.id] = []
                        self.edges[new_node.id].append(branch.id)
        elif isinstance(node_or_chain, TaskChain):
            incoming_chain = node_or_chain
            self.nodes.extend(n for n in incoming_chain.nodes if n not in self.nodes)
            self.edges.update(incoming_chain.edges)
            all_target_ids = {target for targets in incoming_chain.edges.values() for target in targets}
            entry_points.extend([n for n in incoming_chain.nodes if n.id not in all_target_ids])

        if entry_points and previous_leaves:
            for leaf in previous_leaves:
                for entry_point in entry_points:
                    if leaf.id not in self.edges: self.edges[leaf.id] = []
                    if entry_point.id not in self.edges[leaf.id]:
                        self.edges[leaf.id].append(entry_point.id)
        return self
        
    def __rshift__(self, other: Union[TaskNode, Task, 'TaskChain', DecisionFunc, DecisionLLM]) -> 'TaskChain':
        """
        Implements the >> operator to connect this chain with another node, task, or chain.
        
        Args:
            other: The next node, task, or chain to connect
            
        Returns:
            This chain with the new node(s) added
        """
        self.add(other)
        return self


class State(BaseModel):
    """
    Manages the state between task executions in the graph.
    
    Attributes:
        data: Dictionary storing additional data shared across tasks
        task_outputs: Dictionary mapping node IDs to their task outputs
    """
    data: Dict[str, Any] = Field(default_factory=dict)
    task_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    def update(self, node_id: str, output: Any):
        """
        Updates the state with a node's task output.
        
        Args:
            node_id: ID of the node
            output: Output from the task execution
        """
        self.task_outputs[node_id] = output
        
    def get_task_output(self, node_id: str) -> Any:
        """
        Retrieves the output of a specific node's task.
        
        Args:
            node_id: ID of the node
            
        Returns:
            The output of the specified node's task
        """
        return self.task_outputs.get(node_id)
    
    def get_latest_output(self) -> Any:
        """
        Gets the most recent task output.
        
        Returns:
            The output of the most recently executed task
        """
        if not self.task_outputs: return None
        return list(self.task_outputs.values())[-1]


class Graph(BaseModel):
    """
    Main graph structure that manages task execution, state, and workflow.
    
    Attributes:
        default_agent: Default agent to use when a task doesn't specify one
        parallel_execution: Whether to execute independent tasks in parallel
        max_parallel_tasks: Maximum number of tasks to execute in parallel
        show_progress: Whether to display a progress bar during execution
    """
    default_agent: Optional[BaseAgent] = None
    parallel_execution: bool = False
    max_parallel_tasks: int = 4
    show_progress: bool = True
    
    nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = Field(default_factory=list)
    edges: Dict[str, List[str]] = Field(default_factory=dict)
    state: State = Field(default_factory=State)

    storage: Optional[Storage] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        if 'default_agent' in data and data['default_agent'] is not None:
            agent = data['default_agent']
            if not isinstance(agent, BaseAgent):
                 raise TypeError("default_agent must be an instance of a class that inherits from BaseAgent.")
            if not hasattr(agent, 'do_async') or not callable(getattr(agent, 'do_async')):
                raise ValueError("default_agent must have a 'do_async' method.")
        if 'storage' in data:
            self.storage = data.pop('storage')
            if self.storage and not isinstance(self.storage, Storage):
                raise TypeError("storage must be an instance of a class that inherits from Storage.")
        super().__init__(**data)

    def add(self, tasks_chain: Union[Task, TaskNode, TaskChain, DecisionFunc, DecisionLLM]) -> 'Graph':
        """
        Adds a complete workflow (chain) to the graph.
        
        Args:
            tasks_chain: A Task, Node, or fully-formed TaskChain to add to the graph.
            
        Returns:
            This graph for method chaining.
        """
        if not isinstance(tasks_chain, TaskChain):
            tasks_chain = TaskChain().add(tasks_chain)
        
        self.nodes.extend(n for n in tasks_chain.nodes if n not in self.nodes)
        for src, targets in tasks_chain.edges.items():
            if src not in self.edges: self.edges[src] = []
            self.edges[src].extend(t for t in targets if t not in self.edges[src])
        return self
    
    def _get_available_agent(self) -> Any:
        """
        Finds an available agent either from the graph default or from any task node.
        
        Returns:
            An agent that can be used for execution, or None if none is found
        """
        if self.default_agent is not None:
            return self.default_agent
        for node in self.nodes:
            if isinstance(node, TaskNode) and node.task.agent is not None:
                return node.task.agent
        return None

    async def _execute_task(self, node: TaskNode, state: State, verbose: bool = False, *, graph_execution_id: Optional[str] = None) -> Any:
        """
        Executes a single task.
        
        Args:
            node: The TaskNode containing the task to execute
            state: Current state object
            verbose: Whether to print detailed information
            
        Returns:
            The output of the task
        """
        task = node.task
        
        runner = task.agent or self.default_agent
        if runner is None:
            runner = self._get_available_agent()
            if runner is None:
                raise ValueError(f"No agent specified for task '{escape_rich_markup(task.description)}' and no default agent set")
        
        try:

            if verbose:
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{escape_rich_markup(task.description)}[/cyan]")
                runner_type = runner.__class__.__name__ if hasattr(runner, '__class__') else type(runner).__name__
                table.add_row("[bold]Agent:[/bold]", f"[yellow]{escape_rich_markup(runner_type)}[/yellow]")
                if task.tools:
                    tool_names = [escape_rich_markup(t.__class__.__name__ if hasattr(t, '__class__') else str(t)) for t in task.tools]
                    table.add_row("[bold]Tools:[/bold]", f"[green]{escape_rich_markup(', '.join(tool_names))}[/green]")
                panel = Panel(table, title="[bold blue]Upsonic - Executing Task[/bold blue]", border_style="blue", expand=True, width=70)
                console.print(panel)
                spacing()
            
            # Delegate execution to the standardized agent pipeline, passing the state.
            if hasattr(runner, 'do_async'):
                output = await runner.do_async(task, state=state, graph_execution_id=graph_execution_id)
            else:
                output = runner.do(task)
            
            # The task object is now mutated in place by the pipeline.
            if verbose:
                time_taken = task.duration or 0.0
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{escape_rich_markup(task.description)}[/cyan]")
                output_str = self._format_output_for_display(output)
                table.add_row("[bold]Output:[/bold]", f"[green]{output_str}[/green]")
                table.add_row("[bold]Time Taken:[/bold]", f"{time_taken:.2f} seconds")
                if task.total_cost:
                    table.add_row("[bold]Estimated Cost:[/bold]", f"${task.total_cost:.4f}")
                panel = Panel(table, title="[bold green]✅ Task Completed[/bold green]", border_style="green", expand=True, width=70)
                console.print(panel)
                spacing()
            
            return output
            
        except Exception as e:
            if verbose:
                console.print(f"[bold red]Task '{escape_rich_markup(task.description)}' failed: {escape_rich_markup(str(e))}[/bold red]")
            raise
    
    # =======================================================

    async def _evaluate_decision(self, decision_node: Union[DecisionFunc, DecisionLLM], state: State, verbose: bool = False) -> Union[TaskNode, TaskChain, None]:
        """
        Evaluates a decision node to determine which branch to follow.
        
        Args:
            decision_node: The decision node to evaluate
            state: Current state object
            verbose: Whether to print detailed information
            
        Returns:
            The branch to follow (true or false)
        """
        latest_output = state.get_latest_output()
        
        if isinstance(decision_node, DecisionFunc):
            result = decision_node.evaluate(latest_output)
        elif isinstance(decision_node, DecisionLLM):
            agent = self.default_agent or self._get_available_agent()
            if agent is None:
                raise ValueError(f"No agent available for LLM-based decision: '{decision_node.description}'")
            
            prompt = decision_node._generate_prompt(latest_output)
            decision_task = Task(prompt, response_format=DecisionResponse)
            
            # Pass state here as well, in case the decision needs context
            if hasattr(agent, 'do_async'):
                response = await agent.do_async(decision_task, state=state)
            else:
                response = agent.do(decision_task)
            
            result = response.result if hasattr(response, 'result') else False
            
            if verbose:
                console.print(f"[dim]LLM Decision Response: {escape_rich_markup(str(response))}[/dim]")
                console.print(f"[dim]Decision Result: {'Yes' if result else 'No'}[/dim]")
        else:
            raise ValueError(f"Unknown decision node type: {type(decision_node)}")
        
        if verbose:
            table = Table(show_header=False, expand=True, box=None)
            table.add_row("[bold]Decision:[/bold]", f"[cyan]{escape_rich_markup(decision_node.description)}[/cyan]")
            table.add_row("[bold]Result:[/bold]", f"[green]{'Yes' if result else 'No'}[/green]")
            panel = Panel(table, title="[bold yellow]🔀 Evaluating Decision[/bold yellow]", border_style="yellow", expand=True, width=70)
            console.print(panel)
            spacing()
        
        return decision_node.true_branch if result else decision_node.false_branch
    
    def _format_output_for_display(self, output: Any) -> str:
        """
        Format an output value for display in verbose mode.
        
        Args:
            output: The output value to format
            
        Returns:
            A string representation of the output
        """
        if output is None: return ""
        if hasattr(output, '__class__') and hasattr(output.__class__, 'model_dump'):
            try:
                import json
                output_str = json.dumps(output.model_dump(), default=str)
                if len(output_str) > 200: output_str = output_str[:197] + "..."
                return escape_rich_markup(output_str)
            except Exception:
                output_str = str(output)
        else:
            output_str = str(output)
        if len(output_str) > 200: output_str = output_str[:197] + "..."
        return escape_rich_markup(output_str)
    
    def _get_predecessors(self, node: Union[TaskNode, DecisionFunc, DecisionLLM]) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the predecessor nodes that feed into the given node.
        
        Args:
            node: The node to find predecessors for
            
        Returns:
            List of predecessor nodes
        """
        predecessor_ids = {n_id for n_id, next_ids in self.edges.items() if node.id in next_ids}
        return [n for n in self.nodes if n.id in predecessor_ids]
    
    def _get_start_nodes(self) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the starting nodes of the graph (those with no predecessors).
        
        Returns:
            List of start nodes
        """
        all_target_ids = {target_id for targets in self.edges.values() for target_id in targets}
        return [node for node in self.nodes if node.id not in all_target_ids]
    
    def _get_next_nodes(self, node: Union[TaskNode, DecisionFunc, DecisionLLM]) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the nodes that come after the given node.
        
        Args:
            node: The node to find successors for
            
        Returns:
            List of successor nodes
        """
        if node.id not in self.edges: return []
        next_ids = self.edges[node.id]
        return [n for n in self.nodes if n.id in next_ids]

    def _get_all_branch_node_ids(self, branch: Union[TaskNode, TaskChain, DecisionFunc, DecisionLLM, None]) -> Set[str]:
        """Recursively collects all node IDs within a given branch."""
        if not branch: return set()
        ids, queue = set(), [branch]
        while queue:
            current = queue.pop(0)
            if isinstance(current, TaskChain):
                for node in current.nodes:
                    if node.id not in ids:
                        ids.add(node.id)
                        queue.append(node)
            else:
                if current.id not in ids:
                    ids.add(current.id)
                    if isinstance(current, (DecisionFunc, DecisionLLM)):
                        if current.true_branch: queue.append(current.true_branch)
                        if current.false_branch: queue.append(current.false_branch)
        return ids

    async def _run_sequential(self, verbose: bool = False, show_progress: bool = True, *, graph_execution_id: Optional[str] = None) -> State:
        """
        Runs tasks sequentially.
        """
        if verbose:
            console.print(f"[blue]Executing graph with decision support[/blue]")
            spacing()

        start_nodes = self._get_start_nodes()
        execution_queue = list(start_nodes)
        queued_node_ids = {n.id for n in start_nodes}
        executed_node_ids, pruned_node_ids = set(), set()
        all_nodes_count = self._count_all_possible_nodes()
        
        progress_context = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console) if show_progress else None

        if progress_context:
            progress_context.start()
            overall_task = progress_context.add_task("[bold blue]Graph Execution", total=all_nodes_count)

        try:
            while execution_queue:
                node = execution_queue.pop(0)
                if isinstance(node, TaskNode):
                    node.task.context = []
                    predecessors = self._get_predecessors(node)
                    if predecessors:
                        existing_source_ids = {s.task_description_or_id for s in node.task.context if isinstance(s, TaskOutputSource)}
                        for pred in predecessors:
                            if pred.id in executed_node_ids and pred.id not in existing_source_ids:
                                source = TaskOutputSource(task_description_or_id=pred.id)
                                node.task.context.append(source)
                                if verbose:
                                    console.print(f"[dim]Auto-injecting source for node '{pred.id}' into task {escape_rich_markup(node.task.description)}[/dim]")
                    output = await self._execute_task(node, self.state, verbose, graph_execution_id=graph_execution_id)
                    self.state.update(node.id, output)
                    executed_node_ids.add(node.id)
                elif isinstance(node, (DecisionFunc, DecisionLLM)):
                    # Pass the state to the evaluation method
                    branch_to_follow = await self._evaluate_decision(node, self.state, verbose)
                    executed_node_ids.add(node.id)
                    pruned_branch = node.false_branch if branch_to_follow == node.true_branch else node.true_branch
                    pruned_node_ids.update(self._get_all_branch_node_ids(pruned_branch))

                successors = self._get_next_nodes(node)
                for next_node in successors:
                    if next_node.id in queued_node_ids or next_node.id in executed_node_ids or next_node.id in pruned_node_ids:
                        continue
                    if all((p.id in executed_node_ids or p.id in pruned_node_ids) for p in self._get_predecessors(next_node)):
                        execution_queue.append(next_node)
                        queued_node_ids.add(next_node.id)

                if show_progress:
                    completed_count = len(executed_node_ids) + len(pruned_node_ids)
                    if progress_context:
                        progress_context.update(overall_task, completed=completed_count)
        finally:
            if progress_context:
                progress_context.update(overall_task, completed=all_nodes_count)
                progress_context.stop()
        
        if verbose:
            console.print("[bold green]Graph Execution Completed[/bold green]")
            spacing()
        return self.state
    
    def _count_all_possible_nodes(self) -> int:
        """
        Counts all nodes in the graph, which are pre-flattened during construction.
        
        Returns:
            The total number of nodes in the graph.
        """
        return max(len(self.nodes), 1)
    
    async def run_async(self, verbose: bool = True, show_progress: bool = None) -> State:
        """
        Executes the graph, running all tasks in the appropriate order.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar during execution. If None, uses the graph's show_progress attribute.
            
        Returns:
            The final state object with all task outputs
        """
        if show_progress is None: show_progress = self.show_progress
        if verbose:
            console.print("[bold blue]Starting Graph Execution[/bold blue]")
            spacing()
        graph_execution_id = str(uuid.uuid4())

        self.state = State()
        return await self._run_sequential(verbose, show_progress, graph_execution_id=graph_execution_id)

    def run(self, verbose: bool = True, show_progress: bool = None) -> State:
        """
        Executes the graph, running all tasks in the appropriate order synchronously.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar during execution. If None, uses the graph's show_progress attribute.
            
        Returns:
            The final state object with all task outputs
        """
        try: loop = asyncio.get_event_loop()
        except RuntimeError: return asyncio.run(self.run_async(verbose, show_progress))
        if loop.is_running():
            with ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, self.run_async(verbose, show_progress)).result()
        else: return loop.run_until_complete(self.run_async(verbose, show_progress))

    def get_output(self) -> Any:
        """
        Gets the output of the last task executed in the graph.
        
        Returns:
            The output of the last task
        """
        return self.state.get_latest_output()
    
    def get_task_output(self, description: str) -> Any:
        """
        Gets the output of a task by its description.
        
        Args:
            description: The description of the task
            
        Returns:
            The output of the specified task, or None if not found
        """
        for node in self.nodes:
            if isinstance(node, TaskNode) and node.task.description == description:
                output = self.state.get_task_output(node.id)
                if output is not None: return output
        return None

def task(description: str, **kwargs) -> Task:
    """
    Creates a new Task with the given description and parameters.
    
    Args:
        description: The description of the task
        **kwargs: Additional parameters for the Task
        
    Returns:
        A new Task instance
    """
    if 'agent' not in kwargs: kwargs['agent'] = None
    return Task(description=description, **kwargs)

def node(task_instance: Task) -> TaskNode:
    """
    Creates a new TaskNode wrapping the given Task.
    
    Args:
        task_instance: The Task to wrap
        
    Returns:
        A new TaskNode instance
    """
    return TaskNode(task=task_instance)

def create_graph(default_agent: Optional[Any] = None, parallel_execution: bool = False, show_progress: bool = True) -> Graph:
    """
    Creates a new graph with the specified configuration.
    
    Args:
        default_agent: Default agent to use for tasks (AgentConfiguration or Agent)
        parallel_execution: Whether to execute independent tasks in parallel
        show_progress: Whether to display a progress bar during execution
        
    Returns:
        A configured Graph instance
    """
    return Graph(default_agent=default_agent, parallel_execution=parallel_execution, show_progress=show_progress)

def _task_rshift(self, other):
    """
    Implements the >> operator for Task objects to connect them in a chain.
    
    Args:
        other: The next task in the chain
        
    Returns:
        A TaskChain object containing both tasks as nodes
    """
    chain = TaskChain()
    chain.add(self)
    chain.add(other)
    return chain

Task.__rshift__ = _task_rshift