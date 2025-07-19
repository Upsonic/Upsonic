from __future__ import annotations

from typing import List, Dict, Any, Optional, Union, Callable, Set, TYPE_CHECKING
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
import uuid
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.printing import console, spacing, escape_rich_markup
from ..tasks.tasks import Task
from ..tasks.task_response import ObjectResponse
from ..context.sources import TaskOutputSource

from ..direct.base import BaseAgent


class DecisionResponse(ObjectResponse):
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
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, description: str, *, true_branch=None, false_branch=None, id=None, **kwargs):
        """
        Initialize a DecisionLLM with a positional description parameter.
        
        Args:
            description: Human-readable description of the decision
            true_branch: The branch to follow if the LLM decides yes/true
            false_branch: The branch to follow if the LLM decides no/false
            id: Unique identifier for this decision node
        """
        if id is None:
            id = str(uuid.uuid4())
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
        # This is a placeholder - the actual implementation happens 
        # during graph execution using the graph's agent
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
        # If given a Task, wrap it in a TaskNode
        if isinstance(branch, Task):
            branch = TaskNode(task=branch)
            
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
        # If given a Task, wrap it in a TaskNode
        if isinstance(branch, Task):
            branch = TaskNode(task=branch)
            
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
    
    class Config:
        arbitrary_types_allowed = True
    
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
        if id is None:
            id = str(uuid.uuid4())
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
        # If given a Task, wrap it in a TaskNode
        if isinstance(branch, Task):
            branch = TaskNode(task=branch)
            
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
        # If given a Task, wrap it in a TaskNode
        if isinstance(branch, Task):
            branch = TaskNode(task=branch)
            
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
        if not self.nodes:
            return []
        
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
        # Get the current leaf nodes before any modifications.
        previous_leaves = self._get_leaf_nodes()
        
        # Standardize input: If a raw Task is given, wrap it.
        if isinstance(node_or_chain, Task):
            node_or_chain = TaskNode(task=node_or_chain)

        # Add the new content and determine its entry point(s)
        entry_points = []
        if isinstance(node_or_chain, (TaskNode, DecisionFunc, DecisionLLM)):
            new_node = node_or_chain
            if new_node not in self.nodes:
                self.nodes.append(new_node)
            entry_points.append(new_node)

            # If it's a decision node, we must also add its branches to build out the sub-graph.
            if isinstance(new_node, (DecisionFunc, DecisionLLM)):
                for branch in [new_node.true_branch, new_node.false_branch]:
                    if not branch:
                        continue
                    # Add branch content and connect the decision node to the start of the branch.
                    if isinstance(branch, TaskChain):
                        self.nodes.extend(n for n in branch.nodes if n not in self.nodes)
                        self.edges.update(branch.edges)
                        if branch.nodes:
                            if new_node.id not in self.edges: self.edges[new_node.id] = []
                            self.edges[new_node.id].append(branch.nodes[0].id)
                    else:  # TaskNode or another Decision
                        if branch not in self.nodes: self.nodes.append(branch)
                        if new_node.id not in self.edges: self.edges[new_node.id] = []
                        self.edges[new_node.id].append(branch.id)
                        
        elif isinstance(node_or_chain, TaskChain):
            incoming_chain = node_or_chain
            # Add nodes and edges from the incoming chain.
            self.nodes.extend(n for n in incoming_chain.nodes if n not in self.nodes)
            self.edges.update(incoming_chain.edges)
            
            # The entry points are the start nodes of the incoming chain.
            all_target_ids = {target for targets in incoming_chain.edges.values() for target in targets}
            entry_points.extend([n for n in incoming_chain.nodes if n.id not in all_target_ids])

        # Connect the previous leaves to the new entry point(s)
        if entry_points and previous_leaves:
            for leaf in previous_leaves:
                for entry_point in entry_points:
                    if leaf.id not in self.edges:
                        self.edges[leaf.id] = []
                    # Avoid duplicate edges
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
        if not self.task_outputs:
            return None
        
        # Return the most recently added output
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
    # Accept either AgentConfiguration or Direct as the default_agent
    default_agent: Optional[BaseAgent] = None
    parallel_execution: bool = False
    max_parallel_tasks: int = 4
    show_progress: bool = True
    
    # Private attributes (not part of the model schema)
    nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = Field(default_factory=list)
    edges: Dict[str, List[str]] = Field(default_factory=dict)
    state: State = Field(default_factory=State)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        if 'default_agent' in data and data['default_agent'] is not None:
            agent = data['default_agent']
            if not isinstance(agent, BaseAgent):
                 raise TypeError("default_agent must be an instance of a class that inherits from BaseAgent.")
            if not hasattr(agent, 'do_async') or not callable(getattr(agent, 'do_async')):
                raise ValueError("default_agent must have a 'do_async' method.")
        super().__init__(**data)

    def add(self, tasks_chain: Union[Task, TaskNode, TaskChain, DecisionFunc, DecisionLLM]) -> 'Graph':
        """
        Adds a complete workflow (chain) to the graph.
        
        Args:
            tasks_chain: A Task, Node, or fully-formed TaskChain to add to the graph.
            
        Returns:
            This graph for method chaining.
        """
        # If not a chain, create a new one to add to the graph.
        if not isinstance(tasks_chain, TaskChain):
            tasks_chain = TaskChain().add(tasks_chain)
        
        # Merge the chain's nodes and edges into the graph's state
        self.nodes.extend(n for n in tasks_chain.nodes if n not in self.nodes)
        
        # Merge edges carefully to avoid overwriting
        for src, targets in tasks_chain.edges.items():
            if src not in self.edges:
                self.edges[src] = []
            self.edges[src].extend(t for t in targets if t not in self.edges[src])

        return self
    
    def _get_available_agent(self) -> Any:
        """
        Finds an available agent either from the graph default or from any task node.
        
        Returns:
            An agent that can be used for execution, or None if none is found
        """
        # First check if we have a default agent
        if self.default_agent is not None:
            return self.default_agent
        
        # If no default agent, check all task nodes for an agent
        for node in self.nodes:
            if isinstance(node, TaskNode) and node.task.agent is not None:
                return node.task.agent
        
        # No agent found
        return None

    async def _execute_task(self, node: TaskNode, state: State, verbose: bool = False) -> Any:
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
        
        # Use the task's agent or try to find an available agent
        runner = task.agent or self.default_agent
        if runner is None:
            # Try to find any agent from other task nodes
            runner = self._get_available_agent()
            if runner is None:
                raise ValueError(f"No agent specified for task '{task.description}' and no default agent set")
        
        try:
            # Start timing
            start_time = time.time()
            task.start_time = start_time
            
            if verbose:
                # Create and print a task execution panel
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{escape_rich_markup(task.description)}[/cyan]")
                # Display runner type safely
                runner_type = runner.__class__.__name__ if hasattr(runner, '__class__') else type(runner).__name__
                table.add_row("[bold]Agent:[/bold]", f"[yellow]{escape_rich_markup(runner_type)}[/yellow]")
                if task.tools:
                    tool_names = [escape_rich_markup(t.__class__.__name__ if hasattr(t, '__class__') else str(t)) for t in task.tools]
                    table.add_row("[bold]Tools:[/bold]", f"[green]{escape_rich_markup(', '.join(tool_names))}[/green]")
                panel = Panel(
                    table,
                    title="[bold blue]Upsonic - Executing Task[/bold blue]",
                    border_style="blue",
                    expand=True,
                    width=70
                )
                console.print(panel)
                spacing()
            
            
            # Execute the task - use do_async for async execution
            if hasattr(runner, 'do_async'):
                output = await runner.do_async(task, state=state)
            else:
                # Fallback to synchronous do method if do_async is not available
                output = runner.do(task)
            
            # End timing
            end_time = time.time()
            task.end_time = end_time
            
            if verbose:
                # Create and print a task completion panel
                time_taken = end_time - start_time
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{escape_rich_markup(task.description)}[/cyan]")
                
                # Handle different output types for display
                output_str = self._format_output_for_display(output)
                
                table.add_row("[bold]Output:[/bold]", f"[green]{output_str}[/green]")
                table.add_row("[bold]Time Taken:[/bold]", f"{time_taken:.2f} seconds")
                if task.total_cost:
                    table.add_row("[bold]Estimated Cost:[/bold]", f"${task.total_cost:.4f}")
                panel = Panel(
                    table,
                    title="[bold green]✅ Task Completed[/bold green]",
                    border_style="green",
                    expand=True,
                    width=70
                )
                console.print(panel)
                spacing()
            
            return output
            
        except Exception as e:
            if verbose:
                console.print(f"[bold red]Task '{escape_rich_markup(task.description)}' failed: {escape_rich_markup(str(e))}[/bold red]")
            raise
    
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
        # Get the most recent output to evaluate
        latest_output = state.get_latest_output()
        
        # Evaluate differently based on the decision node type
        if isinstance(decision_node, DecisionFunc):
            # For function-based decisions, directly call the function
            result = decision_node.evaluate(latest_output)
        elif isinstance(decision_node, DecisionLLM):
            # For LLM-based decisions, use the default agent or find an available one
            agent = self.default_agent
            if agent is None:
                # Try to find any agent from task nodes
                agent = self._get_available_agent()
                if agent is None:
                    raise ValueError(f"No agent available for LLM-based decision: '{decision_node.description}'")
            
            # Generate the prompt for the LLM
            prompt = decision_node._generate_prompt(latest_output)
            
            # Create a temporary task for the LLM to execute
            decision_task = Task(prompt,
                response_format=DecisionResponse
            )
            
            # Execute the task using the agent
            if hasattr(agent, 'do_async'):
                response = await agent.do_async(decision_task, state=state)
            else:
                # Fallback to synchronous do method if do_async is not available
                response = agent.do(decision_task)
            
            # Get the boolean result from the structured response
            result = response.result if hasattr(response, 'result') else False
            
            if verbose:
                console.print(f"[dim]LLM Decision Response: {escape_rich_markup(str(response))}[/dim]")
                console.print(f"[dim]Decision Result: {'Yes' if result else 'No'}[/dim]")
        else:
            raise ValueError(f"Unknown decision node type: {type(decision_node)}")
        
        if verbose:
            # Create and print a decision evaluation panel
            table = Table(show_header=False, expand=True, box=None)
            table.add_row("[bold]Decision:[/bold]", f"[cyan]{escape_rich_markup(decision_node.description)}[/cyan]")
            table.add_row("[bold]Result:[/bold]", f"[green]{result}[/green]")
            panel = Panel(
                table,
                title="[bold yellow]🔀 Evaluating Decision[/bold yellow]",
                border_style="yellow",
                expand=True,
                width=70
            )
            console.print(panel)
            spacing()
        
        # Return the appropriate branch
        if result:
            return decision_node.true_branch
        else:
            return decision_node.false_branch
    
    def _format_output_for_display(self, output: Any) -> str:
        """
        Format an output value for display in verbose mode.
        
        Args:
            output: The output value to format
            
        Returns:
            A string representation of the output
        """
        # If output is None, return empty string
        if output is None:
            return ""
        
        # If output is a Pydantic model
        if hasattr(output, '__class__') and hasattr(output.__class__, 'model_dump'):
            try:
                # Try to get a compact representation of the model
                model_dict = output.model_dump()
                # Format as compact JSON with max 200 chars
                import json
                output_str = json.dumps(model_dict, default=str)
                if len(output_str) > 200:
                    output_str = output_str[:197] + "..."
                return escape_rich_markup(output_str)
            except Exception:
                # Fallback to str if model_dump fails
                output_str = str(output)
        else:
            # Regular string representation
            output_str = str(output)
        
        # Truncate if too long
        if len(output_str) > 200:
            output_str = output_str[:197] + "..."
            
        return escape_rich_markup(output_str)
    
    def _get_predecessors(self, node: Union[TaskNode, DecisionFunc, DecisionLLM]) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the predecessor nodes that feed into the given node.
        
        Args:
            node: The node to find predecessors for
            
        Returns:
            List of predecessor nodes
        """
        predecessors = []
        # Find all node IDs that have the target node's ID in their edge list
        predecessor_ids = {n_id for n_id, next_ids in self.edges.items() if node.id in next_ids}
        
        # Map IDs back to node objects
        for n in self.nodes:
            if n.id in predecessor_ids:
                predecessors.append(n)
        
        return predecessors
    
    def _get_start_nodes(self) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the starting nodes of the graph (those with no predecessors).
        
        Returns:
            List of start nodes
        """
        # Get all IDs that appear as targets in the edges dictionary
        all_target_ids = {target_id for targets in self.edges.values() for target_id in targets}
        
        # Return nodes that don't appear as targets
        return [node for node in self.nodes if node.id not in all_target_ids]
    
    def _get_next_nodes(self, node: Union[TaskNode, DecisionFunc, DecisionLLM]) -> List[Union[TaskNode, DecisionFunc, DecisionLLM]]:
        """
        Gets the nodes that come after the given node.
        
        Args:
            node: The node to find successors for
            
        Returns:
            List of successor nodes
        """
        next_nodes = []
        
        # Check edges dictionary for successors
        if node.id in self.edges:
            next_ids = self.edges[node.id]
            # Map IDs to node objects
            for next_id in next_ids:
                for n in self.nodes:
                    if n.id == next_id:
                        next_nodes.append(n)
        
        return next_nodes

    def _get_all_branch_node_ids(self, branch: Union[TaskNode, TaskChain, DecisionFunc, DecisionLLM, None]) -> Set[str]:
        """Recursively collects all node IDs within a given branch."""
        if not branch:
            return set()
        
        ids = set()
        queue = [branch]
        
        while queue:
            current = queue.pop(0)
            if isinstance(current, TaskChain):
                for node in current.nodes:
                    if node.id not in ids:
                        ids.add(node.id)
                        queue.append(node)
            else: # TaskNode or Decision
                if current.id not in ids:
                    ids.add(current.id)
                    if isinstance(current, (DecisionFunc, DecisionLLM)):
                        if current.true_branch: queue.append(current.true_branch)
                        if current.false_branch: queue.append(current.false_branch)
        return ids

    async def _run_sequential(self, verbose: bool = False, show_progress: bool = True) -> State:
        """
        Runs tasks sequentially.
        """
        if verbose:
            console.print(f"[blue]Executing graph with decision support[/blue]")
            spacing()
        
        start_nodes = self._get_start_nodes()
        execution_queue = list(start_nodes)
        queued_node_ids = {n.id for n in start_nodes}
        executed_node_ids = set()
        pruned_node_ids = set()

        all_nodes_count = self._count_all_possible_nodes()
        
        progress_context = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) if show_progress else None

        if progress_context:
            progress_context.start()
            overall_task = progress_context.add_task("[bold blue]Graph Execution", total=all_nodes_count)

        try:
            while execution_queue:
                node = execution_queue.pop(0)
                
                if isinstance(node, TaskNode):
                    predecessors = self._get_predecessors(node)
                    if predecessors:
                        existing_source_ids = {s.task_description_or_id for s in node.task.context if isinstance(s, TaskOutputSource)}
                        for pred in predecessors:
                            if pred.id in executed_node_ids and pred.id not in existing_source_ids:
                                source = TaskOutputSource(task_description_or_id=pred.id)
                                node.task.context.append(source)
                                if verbose:
                                    console.print(f"[dim]Auto-injecting output of node '{pred.id}' into task {escape_rich_markup(node.task.description)}[/dim]")
                    
                    output = await self._execute_task(node, self.state, verbose)
                    self.state.update(node.id, output)
                    executed_node_ids.add(node.id)

                elif isinstance(node, (DecisionFunc, DecisionLLM)):
                    branch_to_follow = await self._evaluate_decision(node, self.state, verbose)
                    executed_node_ids.add(node.id)

                    pruned_branch = node.false_branch if branch_to_follow == node.true_branch else node.true_branch
                    pruned_node_ids.update(self._get_all_branch_node_ids(pruned_branch))

                # After node completion, check all successors to see if they are ready to be queued.
                successors = self._get_next_nodes(node)
                for next_node in successors:
                    if next_node.id in queued_node_ids or next_node.id in executed_node_ids:
                        continue
                    
                    # A node cannot be queued if it has been marked for pruning.
                    if next_node.id in pruned_node_ids:
                        continue

                    predecessors = self._get_predecessors(next_node)
                    is_ready = all((p.id in executed_node_ids or p.id in pruned_node_ids) for p in predecessors)
                    
                    if is_ready:
                        execution_queue.append(next_node)
                        queued_node_ids.add(next_node.id)

                if show_progress:
                    completed_count = len(executed_node_ids) + len(pruned_node_ids)
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
        # Use class attribute if show_progress is not explicitly specified
        if show_progress is None:
            show_progress = self.show_progress
            
        if verbose:
            console.print("[bold blue]Starting Graph Execution[/bold blue]")
            spacing()
        
        # Reset state
        self.state = State()
        
        # With decision support, we always use the sequential implementation for now
        return await self._run_sequential(verbose, show_progress)

    def run(self, verbose: bool = True, show_progress: bool = None) -> State:
        """
        Executes the graph, running all tasks in the appropriate order synchronously.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar during execution. If None, uses the graph's show_progress attribute.
            
        Returns:
            The final state object with all task outputs
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self.run_async(verbose, show_progress))
        
        if loop.is_running():
            # Event loop is already running, we need to run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_async(verbose, show_progress))
                return future.result()
        else:
            # Event loop exists but not running, we can use it
            return loop.run_until_complete(self.run_async(verbose, show_progress))

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
                if output is not None:
                    return output
        
        return None


# Helper functions to work with the existing Task class
def task(description: str, **kwargs) -> Task:
    """
    Creates a new Task with the given description and parameters.
    
    Args:
        description: The description of the task
        **kwargs: Additional parameters for the Task
        
    Returns:
        A new Task instance
    """
    # Ensure agent is explicitly set to None if not provided
    if 'agent' not in kwargs:
        kwargs['agent'] = None
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

def create_graph(default_agent: Optional[Any] = None,
                 parallel_execution: bool = False,
                 show_progress: bool = True) -> Graph:
    """
    Creates a new graph with the specified configuration.
    
    Args:
        default_agent: Default agent to use for tasks (AgentConfiguration or Direct)
        parallel_execution: Whether to execute independent tasks in parallel
        show_progress: Whether to display a progress bar during execution
        
    Returns:
        A configured Graph instance
    """
    return Graph(
        default_agent=default_agent,
        parallel_execution=parallel_execution,
        show_progress=show_progress
    )


# Enable Task objects to use the >> operator directly
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

# Apply the patch to the Task class
Task.__rshift__ = _task_rshift