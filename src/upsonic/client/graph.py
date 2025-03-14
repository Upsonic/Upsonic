from typing import List, Dict, Any, Optional, Union, Callable, Set
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .printing import console, spacing
from .tasks.tasks import Task
from .agent_configuration.agent_configuration import AgentConfiguration

# Import Direct for type checking
try:
    from .direct_llm.direct import Direct
except ImportError:
    # Define a placeholder for type checking if the import fails
    class Direct:
        pass


class TaskNode(BaseModel):
    """
    Wrapper around a Task that adds graph connectivity features.
    
    Attributes:
        task: The Task object this node wraps
        id: Unique identifier for this node
    """
    task: Task
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # For edge management
    next_nodes: List['TaskNode'] = Field(default_factory=list)
    
    def __rshift__(self, other: Union['TaskNode', Task]) -> 'TaskChain':
        """
        Implements the >> operator to connect nodes in a chain.
        
        Args:
            other: The next node or task in the chain
            
        Returns:
            A TaskChain object containing both nodes
        """
        chain = TaskChain()
        chain.add(self)
        
        # If the other object is a Task, wrap it in a TaskNode
        if isinstance(other, Task):
            other = TaskNode(task=other)
            
        chain.add(other)
        return chain


class TaskChain:
    """
    Represents a chain of connected task nodes.
    
    Attributes:
        nodes: List of nodes in the chain
        edges: Dictionary mapping node IDs to their next nodes
    """
    def __init__(self):
        self.nodes: List[TaskNode] = []
        self.edges: Dict[str, List[str]] = {}
        
    def add(self, node: Union[TaskNode, Task, 'TaskChain']) -> 'TaskChain':
        """
        Adds a node or another chain to this chain.
        
        Args:
            node: The node, task, or chain to add
            
        Returns:
            This chain for method chaining
        """
        # If given a Task, wrap it in a TaskNode
        if isinstance(node, Task):
            node = TaskNode(task=node)
        
        if isinstance(node, TaskNode):
            if self.nodes:
                last_node = self.nodes[-1]
                if last_node.id not in self.edges:
                    self.edges[last_node.id] = []
                self.edges[last_node.id].append(node.id)
                last_node.next_nodes.append(node)
            self.nodes.append(node)
        elif isinstance(node, TaskChain):
            if self.nodes and node.nodes:
                last_node = self.nodes[-1]
                first_of_new = node.nodes[0]
                if last_node.id not in self.edges:
                    self.edges[last_node.id] = []
                self.edges[last_node.id].append(first_of_new.id)
                last_node.next_nodes.append(first_of_new)
            
            # Merge the other chain's edges into this one
            self.edges.update(node.edges)
            self.nodes.extend(node.nodes)
            
        return self
        
    def __rshift__(self, other: Union[TaskNode, Task, 'TaskChain']) -> 'TaskChain':
        """
        Implements the >> operator to connect this chain with another node, task, or chain.
        
        Args:
            other: The next node, task, or chain to connect
            
        Returns:
            This chain with the new node(s) added
        """
        return self.add(other)


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
    default_agent: Optional[Any] = None
    parallel_execution: bool = False
    max_parallel_tasks: int = 4
    show_progress: bool = True
    
    # Private attributes (not part of the model schema)
    nodes: List[TaskNode] = Field(default_factory=list)
    edges: Dict[str, List[str]] = Field(default_factory=dict)
    state: State = Field(default_factory=State)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        # Validate that default_agent is either AgentConfiguration or Direct
        if 'default_agent' in data and data['default_agent'] is not None:
            agent = data['default_agent']
            # Check if it has the 'do' method which both types should have
            if not hasattr(agent, 'do') or not callable(getattr(agent, 'do')):
                raise ValueError("default_agent must be an instance of AgentConfiguration or Direct with a 'do' method")
        super().__init__(**data)
    
    def add(self, tasks_chain: Union[Task, TaskNode, TaskChain]) -> 'Graph':
        """
        Adds tasks to the graph.
        
        Args:
            tasks_chain: A Task, TaskNode, or TaskChain to add to the graph
            
        Returns:
            This graph for method chaining
        """
        # If given a Task, wrap it in a TaskNode
        if isinstance(tasks_chain, Task):
            tasks_chain = TaskNode(task=tasks_chain)
        
        if isinstance(tasks_chain, TaskNode):
            self.nodes.append(tasks_chain)
        elif isinstance(tasks_chain, TaskChain):
            self.nodes.extend(tasks_chain.nodes)
            self.edges.update(tasks_chain.edges)
            
        return self
    
    def _execute_task(self, node: TaskNode, state: State, verbose: bool = False) -> Any:
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
        
        # Use the task's agent or the graph's default agent
        runner = task.agent or self.default_agent
        if runner is None:
            raise ValueError(f"No agent specified for task '{task.description}' and no default agent set")
        
        try:
            # Start timing
            start_time = time.time()
            task.start_time = start_time
            
            if verbose:
                # Create and print a task execution panel
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{task.description}[/cyan]")
                # Display runner type safely
                runner_type = runner.__class__.__name__ if hasattr(runner, '__class__') else type(runner).__name__
                table.add_row("[bold]Agent:[/bold]", f"[yellow]{runner_type}[/yellow]")
                if task.tools:
                    tool_names = [t.__class__.__name__ if hasattr(t, '__class__') else str(t) for t in task.tools]
                    table.add_row("[bold]Tools:[/bold]", f"[green]{', '.join(tool_names)}[/green]")
                panel = Panel(
                    table,
                    title="[bold blue]Upsonic - Executing Task[/bold blue]",
                    border_style="blue",
                    expand=True,
                    width=70
                )
                console.print(panel)
                spacing()
            
            # Get previous outputs if available
            previous_outputs = [state.get_task_output(prev_node.id) for prev_node in self._get_predecessors(node)]
            
            # Add previous outputs as context if appropriate
            if previous_outputs and not task.context:
                task.context = previous_outputs
            
            # Execute the task - both AgentConfiguration and Direct have the do method
            output = runner.do(task)
            
            # End timing
            end_time = time.time()
            task.end_time = end_time
            
            if verbose:
                # Create and print a task completion panel
                time_taken = end_time - start_time
                table = Table(show_header=False, expand=True, box=None)
                table.add_row("[bold]Task:[/bold]", f"[cyan]{task.description}[/cyan]")
                
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
                console.print(f"[bold red]Task '{task.description}' failed: {str(e)}[/bold red]")
            raise
    
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
                return output_str
            except Exception:
                # Fallback to str if model_dump fails
                output_str = str(output)
        else:
            # Regular string representation
            output_str = str(output)
        
        # Truncate if too long
        if len(output_str) > 200:
            output_str = output_str[:197] + "..."
            
        return output_str
    
    def _get_predecessors(self, node: TaskNode) -> List[TaskNode]:
        """
        Gets the predecessor nodes that feed into the given node.
        
        Args:
            node: The node to find predecessors for
            
        Returns:
            List of predecessor nodes
        """
        predecessors = []
        for n in self.nodes:
            for next_node in n.next_nodes:
                if next_node.id == node.id:
                    predecessors.append(n)
        return predecessors
    
    def _get_start_nodes(self) -> List[TaskNode]:
        """
        Gets the starting nodes of the graph (those with no predecessors).
        
        Returns:
            List of start nodes
        """
        all_next_ids = {next_id for node in self.nodes for next_node in node.next_nodes for next_id in [next_node.id]}
        return [node for node in self.nodes if node.id not in all_next_ids]
    
    def _get_next_nodes(self, node: TaskNode) -> List[TaskNode]:
        """
        Gets the nodes that come after the given node.
        
        Args:
            node: The node to find successors for
            
        Returns:
            List of successor nodes
        """
        return node.next_nodes
    
    def _topological_sort(self) -> List[TaskNode]:
        """
        Performs a topological sort of the graph to determine execution order.
        
        Returns:
            List of nodes in execution order
        """
        # Find all nodes with no incoming edges
        start_nodes = self._get_start_nodes()
        
        # Initialize the result list
        result = []
        
        # Initialize sets for temporary and permanent marks
        temp_marks: Set[str] = set()
        perm_marks: Set[str] = set()
        
        def visit(node: TaskNode):
            if node.id in perm_marks:
                return
            if node.id in temp_marks:
                raise ValueError("Graph has cycles, which are not supported")
            
            temp_marks.add(node.id)
            
            for next_node in self._get_next_nodes(node):
                visit(next_node)
            
            temp_marks.remove(node.id)
            perm_marks.add(node.id)
            result.append(node)
        
        # Visit all unmarked nodes
        for node in start_nodes:
            if node.id not in perm_marks:
                visit(node)
        
        # Reverse to get correct order
        return list(reversed(result))
    
    def run(self, verbose: bool = False, show_progress: bool = None) -> State:
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
        
        if self.parallel_execution:
            return self._run_parallel(verbose, show_progress)
        else:
            return self._run_sequential(verbose, show_progress)
    
    def _run_sequential(self, verbose: bool = False, show_progress: bool = True) -> State:
        """
        Runs tasks sequentially in topological order.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar
            
        Returns:
            The final state object
        """
        # Get nodes in execution order
        execution_order = self._topological_sort()
        
        if verbose:
            console.print(f"[blue]Executing {len(execution_order)} tasks sequentially[/blue]")
            spacing()
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                overall_task = progress.add_task(f"[bold blue]Overall Progress ({len(execution_order)} tasks)", total=len(execution_order))
                
                # Execute each task
                for node in execution_order:
                    # Update task description to show current task
                    desc = f"[bold blue]Overall Progress ({len(execution_order)} tasks) - Current: [cyan]{node.task.description[:40]}{'...' if len(node.task.description) > 40 else ''}[/cyan]"
                    progress.update(overall_task, description=desc)
                    
                    output = self._execute_task(node, self.state, verbose)
                    self.state.update(node.id, output)
                    
                    progress.update(overall_task, advance=1)
        else:
            # Execute each task without progress bar
            for node in execution_order:
                output = self._execute_task(node, self.state, verbose)
                self.state.update(node.id, output)
        
        if verbose:
            console.print("[bold green]Graph Execution Completed[/bold green]")
            spacing()
        
        return self.state
    
    def _run_parallel(self, verbose: bool = False, show_progress: bool = True) -> State:
        """
        Runs tasks in parallel where possible.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar
            
        Returns:
            The final state object
        """
        # Get nodes in execution order
        execution_order = self._topological_sort()
        
        if verbose:
            console.print(f"[blue]Executing {len(execution_order)} tasks with parallel execution where possible[/blue]")
            spacing()
        
        # Track completed tasks
        completed_tasks: Set[str] = set()
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                overall_task = progress.add_task(f"[bold blue]Overall Progress ({len(execution_order)} tasks)", total=len(execution_order))
                
                # Continue until all tasks are executed
                while len(completed_tasks) < len(execution_order):
                    # Find tasks that are ready to execute (all predecessors completed)
                    ready_tasks = []
                    for node in execution_order:
                        if node.id in completed_tasks:
                            continue
                        
                        predecessors = self._get_predecessors(node)
                        if all(pred.id in completed_tasks for pred in predecessors):
                            ready_tasks.append(node)
                    
                    if not ready_tasks:
                        raise ValueError("No tasks ready to execute but not all tasks completed - possible cycle in graph")
                    
                    # Limit the number of parallel tasks
                    batch = ready_tasks[:self.max_parallel_tasks]
                    
                    if verbose:
                        console.print(f"[blue]Executing batch of {len(batch)} tasks in parallel[/blue]")
                    
                    # Update progress description to show current batch
                    batch_desc = ", ".join([node.task.description[:20] + ("..." if len(node.task.description) > 20 else "") for node in batch[:3]])
                    if len(batch) > 3:
                        batch_desc += f" and {len(batch) - 3} more"
                    progress.update(overall_task, description=f"[bold blue]Overall Progress ({len(execution_order)} tasks) - Current batch: [cyan]{batch_desc}[/cyan]")
                    
                    # Execute batch in parallel
                    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                        future_to_node = {
                            executor.submit(self._execute_task, node, self.state, verbose): node 
                            for node in batch
                        }
                        
                        for future in as_completed(future_to_node):
                            node = future_to_node[future]
                            try:
                                output = future.result()
                                self.state.update(node.id, output)
                                completed_tasks.add(node.id)
                                
                                # Update overall progress
                                progress.update(overall_task, advance=1)
                            except Exception as e:
                                if verbose:
                                    console.print(f"[bold red]Task '{node.task.description}' failed: {str(e)}[/bold red]")
                                raise
        else:
            # Continue until all tasks are executed without progress bar
            while len(completed_tasks) < len(execution_order):
                # Find tasks that are ready to execute (all predecessors completed)
                ready_tasks = []
                for node in execution_order:
                    if node.id in completed_tasks:
                        continue
                    
                    predecessors = self._get_predecessors(node)
                    if all(pred.id in completed_tasks for pred in predecessors):
                        ready_tasks.append(node)
                
                if not ready_tasks:
                    raise ValueError("No tasks ready to execute but not all tasks completed - possible cycle in graph")
                
                # Limit the number of parallel tasks
                batch = ready_tasks[:self.max_parallel_tasks]
                
                if verbose:
                    console.print(f"[blue]Executing batch of {len(batch)} tasks in parallel[/blue]")
                
                # Execute batch in parallel
                with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                    future_to_node = {
                        executor.submit(self._execute_task, node, self.state, verbose): node 
                        for node in batch
                    }
                    
                    for future in as_completed(future_to_node):
                        node = future_to_node[future]
                        try:
                            output = future.result()
                            self.state.update(node.id, output)
                            completed_tasks.add(node.id)
                        except Exception as e:
                            if verbose:
                                console.print(f"[bold red]Task '{node.task.description}' failed: {str(e)}[/bold red]")
                            raise
        
        if verbose:
            console.print("[bold green]Graph Execution Completed[/bold green]")
            spacing()
        
        return self.state
    
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
            if node.task.description == description:
                return self.state.get_task_output(node.id)
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
    chain.add(TaskNode(task=self))
    
    if isinstance(other, Task):
        chain.add(TaskNode(task=other))
    elif isinstance(other, TaskNode):
        chain.add(other)
    elif isinstance(other, TaskChain):
        chain = chain.add(other)
    
    return chain

# Apply the patch to the Task class
Task.__rshift__ = _task_rshift
