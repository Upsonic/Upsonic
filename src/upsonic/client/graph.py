from typing import List, Dict, Any, Optional, Union, Callable, Set
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .printing import console, spacing, escape_rich_markup
from .tasks.tasks import Task
from .tasks.task_response import ObjectResponse
from .agent_configuration.agent_configuration import AgentConfiguration

# Define DecisionResponse at module level
class DecisionResponse(ObjectResponse):
    """Response type for LLM-based decisions that returns a boolean result."""
    result: bool

# Import Direct for type checking
try:
    from .direct_llm.direct import Direct
except ImportError:
    # Define a placeholder for type checking if the import fails
    class Direct:
        pass


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
    
    def evaluate(self, data: Any) -> bool:
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
        Implements the >> operator to chain this decision with another node.
        Both the true and false branches will converge to the specified node.
        
        Args:
            other: The node, task, or chain to connect after both branches
            
        Returns:
            A TaskChain containing the decision and its branches
        """
        chain = TaskChain()
        chain.nodes.append(self)
        
        # Add the next node/chain
        if isinstance(other, Task):
            other_node = TaskNode(task=other)
            chain.nodes.append(other_node)
        elif isinstance(other, TaskNode):
            chain.nodes.append(other)
        elif isinstance(other, TaskChain):
            chain.nodes.extend(other.nodes)
        elif isinstance(other, (DecisionFunc, DecisionLLM)):
            chain.nodes.append(other)
            
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
        Implements the >> operator to chain this decision with another node.
        Both the true and false branches will converge to the specified node.
        
        Args:
            other: The node, task, or chain to connect after both branches
            
        Returns:
            A TaskChain containing the decision and its branches
        """
        chain = TaskChain()
        chain.nodes.append(self)
        
        # Add the next node/chain
        if isinstance(other, Task):
            other_node = TaskNode(task=other)
            chain.nodes.append(other_node)
        elif isinstance(other, TaskNode):
            chain.nodes.append(other)
        elif isinstance(other, TaskChain):
            chain.nodes.extend(other.nodes)
        elif isinstance(other, (DecisionFunc, DecisionLLM)):
            chain.nodes.append(other)
            
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
    
    # For edge management
    next_nodes: List['TaskNode'] = Field(default_factory=list)
    
    def __rshift__(self, other: Union['TaskNode', Task, 'DecisionFunc', 'DecisionLLM']) -> 'TaskChain':
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
        self.nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = []
        self.edges: Dict[str, List[str]] = {}
        
    def add(self, node: Union[TaskNode, Task, 'TaskChain', DecisionFunc, DecisionLLM]) -> 'TaskChain':
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
                
                # If the last node is a TaskNode, update its next_nodes
                if isinstance(last_node, TaskNode):
                    last_node.next_nodes.append(node)
            self.nodes.append(node)
        elif isinstance(node, (DecisionFunc, DecisionLLM)):
            # Add the decision node itself
            self.nodes.append(node)
            
            # Connect decision node to its branches
            if node.true_branch:
                if isinstance(node.true_branch, TaskNode):
                    if node.id not in self.edges:
                        self.edges[node.id] = []
                    self.edges[node.id].append(node.true_branch.id)
                    
                    # Add true branch node if not already in the chain
                    if node.true_branch not in self.nodes:
                        self.nodes.append(node.true_branch)
                elif isinstance(node.true_branch, TaskChain):
                    # Add all nodes from the true branch chain
                    if node.true_branch.nodes:
                        first_node = node.true_branch.nodes[0]
                        if node.id not in self.edges:
                            self.edges[node.id] = []
                        self.edges[node.id].append(first_node.id)
                        
                        # Add the nodes from the true branch
                        for n in node.true_branch.nodes:
                            if n not in self.nodes:
                                self.nodes.append(n)
                        
                        # Merge the edges from the true branch
                        for src, targets in node.true_branch.edges.items():
                            if src not in self.edges:
                                self.edges[src] = []
                            self.edges[src].extend(targets)
            
            # Connect false branch as well
            if node.false_branch:
                if isinstance(node.false_branch, TaskNode):
                    if node.id not in self.edges:
                        self.edges[node.id] = []
                    self.edges[node.id].append(node.false_branch.id)
                    
                    # Add false branch node if not already in the chain
                    if node.false_branch not in self.nodes:
                        self.nodes.append(node.false_branch)
                elif isinstance(node.false_branch, TaskChain):
                    # Add all nodes from the false branch chain
                    if node.false_branch.nodes:
                        first_node = node.false_branch.nodes[0]
                        if node.id not in self.edges:
                            self.edges[node.id] = []
                        self.edges[node.id].append(first_node.id)
                        
                        # Add the nodes from the false branch
                        for n in node.false_branch.nodes:
                            if n not in self.nodes:
                                self.nodes.append(n)
                        
                        # Merge the edges from the false branch
                        for src, targets in node.false_branch.edges.items():
                            if src not in self.edges:
                                self.edges[src] = []
                            self.edges[src].extend(targets)
        elif isinstance(node, TaskChain):
            if self.nodes and node.nodes:
                last_node = self.nodes[-1]
                first_of_new = node.nodes[0]
                if last_node.id not in self.edges:
                    self.edges[last_node.id] = []
                self.edges[last_node.id].append(first_of_new.id)
                
                # If the last node is a TaskNode, update its next_nodes
                if isinstance(last_node, TaskNode) and isinstance(first_of_new, TaskNode):
                    last_node.next_nodes.append(first_of_new)
            
            # Merge the other chain's edges into this one
            self.edges.update(node.edges)
            self.nodes.extend(node.nodes)
            
        return self
        
    def __rshift__(self, other: Union[TaskNode, Task, 'TaskChain', DecisionFunc, DecisionLLM]) -> 'TaskChain':
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
    nodes: List[Union[TaskNode, DecisionFunc, DecisionLLM]] = Field(default_factory=list)
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
    
    def add(self, tasks_chain: Union[Task, TaskNode, TaskChain, DecisionFunc, DecisionLLM]) -> 'Graph':
        """
        Adds tasks to the graph.
        
        Args:
            tasks_chain: A Task, TaskNode, TaskChain, DecisionFunc, or DecisionLLM to add to the graph
            
        Returns:
            This graph for method chaining
        """
        # If given a Task, wrap it in a TaskNode
        if isinstance(tasks_chain, Task):
            tasks_chain = TaskNode(task=tasks_chain)
        
        if isinstance(tasks_chain, TaskNode):
            self.nodes.append(tasks_chain)
        elif isinstance(tasks_chain, (DecisionFunc, DecisionLLM)):
            self.nodes.append(tasks_chain)
            
            # Handle decision branches
            if tasks_chain.true_branch:
                if isinstance(tasks_chain.true_branch, TaskNode):
                    if tasks_chain.true_branch not in self.nodes:
                        self.nodes.append(tasks_chain.true_branch)
                    if tasks_chain.id not in self.edges:
                        self.edges[tasks_chain.id] = []
                    self.edges[tasks_chain.id].append(tasks_chain.true_branch.id)
                elif isinstance(tasks_chain.true_branch, TaskChain):
                    # Add all nodes and edges from the true branch chain
                    for node in tasks_chain.true_branch.nodes:
                        if node not in self.nodes:
                            self.nodes.append(node)
                    
                    # Add the connection from decision to first node of true branch
                    if tasks_chain.true_branch.nodes:
                        first_node = tasks_chain.true_branch.nodes[0]
                        if tasks_chain.id not in self.edges:
                            self.edges[tasks_chain.id] = []
                        self.edges[tasks_chain.id].append(first_node.id)
                    
                    # Merge edges from the true branch
                    for src, targets in tasks_chain.true_branch.edges.items():
                        if src not in self.edges:
                            self.edges[src] = []
                        self.edges[src].extend(targets)
            
            # Handle false branch
            if tasks_chain.false_branch:
                if isinstance(tasks_chain.false_branch, TaskNode):
                    if tasks_chain.false_branch not in self.nodes:
                        self.nodes.append(tasks_chain.false_branch)
                    if tasks_chain.id not in self.edges:
                        self.edges[tasks_chain.id] = []
                    self.edges[tasks_chain.id].append(tasks_chain.false_branch.id)
                elif isinstance(tasks_chain.false_branch, TaskChain):
                    # Add all nodes and edges from the false branch chain
                    for node in tasks_chain.false_branch.nodes:
                        if node not in self.nodes:
                            self.nodes.append(node)
                    
                    # Add the connection from decision to first node of false branch
                    if tasks_chain.false_branch.nodes:
                        first_node = tasks_chain.false_branch.nodes[0]
                        if tasks_chain.id not in self.edges:
                            self.edges[tasks_chain.id] = []
                        self.edges[tasks_chain.id].append(first_node.id)
                    
                    # Merge edges from the false branch
                    for src, targets in tasks_chain.false_branch.edges.items():
                        if src not in self.edges:
                            self.edges[src] = []
                        self.edges[src].extend(targets)
                    
        elif isinstance(tasks_chain, TaskChain):
            self.nodes.extend(tasks_chain.nodes)
            self.edges.update(tasks_chain.edges)
            
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
    
    def _evaluate_decision(self, decision_node: Union[DecisionFunc, DecisionLLM], state: State, verbose: bool = False) -> Union[TaskNode, TaskChain, None]:
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
        # Check node connections in the edges dictionary
        for n_id, next_ids in self.edges.items():
            if node.id in next_ids:
                # Find the node object
                for n in self.nodes:
                    if n.id == n_id:
                        predecessors.append(n)
                        break
        
        # Also check TaskNode's next_nodes connections
        for n in self.nodes:
            if isinstance(n, TaskNode):
                for next_node in n.next_nodes:
                    if next_node.id == node.id:
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
        
        # Also get all IDs that appear as targets in TaskNode.next_nodes
        for n in self.nodes:
            if isinstance(n, TaskNode):
                for next_node in n.next_nodes:
                    all_target_ids.add(next_node.id)
        
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
        
        # Check edges dictionary
        if node.id in self.edges:
            next_ids = self.edges[node.id]
            for next_id in next_ids:
                for n in self.nodes:
                    if n.id == next_id:
                        next_nodes.append(n)
        
        # If it's a TaskNode, also check its next_nodes attribute
        if isinstance(node, TaskNode):
            for next_node in node.next_nodes:
                if next_node not in next_nodes:  # Avoid duplicates
                    next_nodes.append(next_node)
        
        return next_nodes
    
    def _run_sequential(self, verbose: bool = False, show_progress: bool = True) -> State:
        """
        Runs tasks sequentially with support for decision nodes.
        
        Args:
            verbose: Whether to print detailed information
            show_progress: Whether to display a progress bar
            
        Returns:
            The final state object
        """
        if verbose:
            console.print(f"[blue]Executing graph with decision support[/blue]")
            spacing()
        
        # We can't use topological sort with decisions, so we use a dynamic execution approach
        execution_queue = self._get_start_nodes()
        executed_nodes = set()
        
        # Count all possible nodes in the graph including all branches
        all_nodes = self._count_all_possible_nodes()
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                # Set total to all possible nodes
                overall_task = progress.add_task("[bold blue]Graph Execution", total=all_nodes)
                
                # Process nodes until the queue is empty
                completed_nodes = 0
                while execution_queue:
                    node = execution_queue.pop(0)
                    
                    # Update progress description
                    if isinstance(node, TaskNode):
                        desc = f"[bold blue]Graph Execution - Current: [cyan]{escape_rich_markup(node.task.description[:40])}{'...' if len(node.task.description) > 40 else ''}[/cyan]"
                    else:
                        desc = f"[bold blue]Graph Execution - Evaluating Decision: [yellow]{escape_rich_markup(node.description)}[/yellow]"
                    
                    # Update the progress bar
                    progress.update(overall_task, description=desc)
                    
                    # Process the node
                    if isinstance(node, TaskNode):
                        # Get the latest output for debugging
                        if verbose:
                            latest_output = self.state.get_latest_output()
                            if latest_output:
                                console.print(f"[dim]********latest_output[/dim]")
                                console.print(f"[dim]{escape_rich_markup(str(latest_output))}[/dim]")
                                
                        # If this task doesn't have context but there are previous outputs, add them as context
                        if not node.task.context and self.state.task_outputs:
                            # Get the latest output to use as context
                            latest_output = self.state.get_latest_output()
                            if latest_output:
                                # Set the context for this task
                                node.task.context = [latest_output]
                                if verbose:
                                    console.print(f"[dim]Setting context from previous output for task: {escape_rich_markup(node.task.description)}[/dim]")
                        
                        output = self._execute_task(node, self.state, verbose)
                        self.state.update(node.id, output)
                        executed_nodes.add(node.id)
                        
                        # Add successor nodes to the queue
                        for next_node in self._get_next_nodes(node):
                            # Only add if all predecessors have been executed
                            predecessors = self._get_predecessors(next_node)
                            if all(pred.id in executed_nodes for pred in predecessors):
                                execution_queue.append(next_node)
                    
                    elif isinstance(node, (DecisionFunc, DecisionLLM)):
                        # Evaluate the decision
                        branch = self._evaluate_decision(node, self.state, verbose)
                        executed_nodes.add(node.id)
                        
                        # Add the appropriate branch to the execution queue
                        if branch:
                            if isinstance(branch, TaskNode):
                                # Make sure the task in the branch has context from previous outputs
                                if not branch.task.context and self.state.task_outputs:
                                    latest_output = self.state.get_latest_output()
                                    if latest_output:
                                        branch.task.context = [latest_output]
                                        if verbose:
                                            console.print(f"[dim]Setting context from previous output for branch task: {escape_rich_markup(branch.task.description)}[/dim]")
                                
                                execution_queue.insert(0, branch)
                            elif isinstance(branch, TaskChain):
                                # Add all nodes from the branch to the queue in reverse order
                                branch_nodes = list(reversed(branch.nodes))
                                
                                # Set context for the first task in the branch if it doesn't have context
                                if branch_nodes and isinstance(branch_nodes[-1], TaskNode):
                                    first_task = branch_nodes[-1]
                                    if not first_task.task.context and self.state.task_outputs:
                                        latest_output = self.state.get_latest_output()
                                        if latest_output:
                                            first_task.task.context = [latest_output]
                                            if verbose:
                                                console.print(f"[dim]Setting context from previous output for first branch chain task: {escape_rich_markup(first_task.task.description)}[/dim]")
                                
                                for branch_node in branch_nodes:
                                    execution_queue.insert(0, branch_node)
                            elif isinstance(branch, (DecisionFunc, DecisionLLM)):
                                # Handle the case where a decision returns another decision node
                                # Insert at the front of the queue to be processed next
                                if verbose:
                                    console.print(f"[dim]Decision returned another decision node: {escape_rich_markup(branch.description)}[/dim]")
                                execution_queue.insert(0, branch)
                    
                    # Increment completed nodes and update progress
                    completed_nodes += 1
                    progress.update(overall_task, completed=completed_nodes)
                
                # Make sure the progress bar reaches 100% at the end
                progress.update(overall_task, completed=all_nodes)
        else:
            # Process nodes without progress bar
            while execution_queue:
                node = execution_queue.pop(0)
                
                # Process the node
                if isinstance(node, TaskNode):
                    # If this task doesn't have context but there are previous outputs, add them as context
                    if not node.task.context and self.state.task_outputs:
                        # Get the latest output to use as context
                        latest_output = self.state.get_latest_output()
                        if latest_output:
                            # Set the context for this task
                            node.task.context = [latest_output]
                    
                    output = self._execute_task(node, self.state, verbose)
                    self.state.update(node.id, output)
                    executed_nodes.add(node.id)
                    
                    # Add successor nodes to the queue
                    for next_node in self._get_next_nodes(node):
                        # Only add if all predecessors have been executed
                        predecessors = self._get_predecessors(next_node)
                        if all(pred.id in executed_nodes for pred in predecessors):
                            execution_queue.append(next_node)
                
                elif isinstance(node, (DecisionFunc, DecisionLLM)):
                    # Evaluate the decision
                    branch = self._evaluate_decision(node, self.state, verbose)
                    executed_nodes.add(node.id)
                    
                    # Add the appropriate branch to the execution queue
                    if branch:
                        if isinstance(branch, TaskNode):
                            # Make sure the task in the branch has context from previous outputs
                            if not branch.task.context and self.state.task_outputs:
                                latest_output = self.state.get_latest_output()
                                if latest_output:
                                    branch.task.context = [latest_output]
                                    
                            execution_queue.insert(0, branch)
                        elif isinstance(branch, TaskChain):
                            # Add all nodes from the branch to the queue in reverse order
                            branch_nodes = list(reversed(branch.nodes))
                            
                            # Set context for the first task in the branch if it doesn't have context
                            if branch_nodes and isinstance(branch_nodes[-1], TaskNode):
                                first_task = branch_nodes[-1]
                                if not first_task.task.context and self.state.task_outputs:
                                    latest_output = self.state.get_latest_output()
                                    if latest_output:
                                        first_task.task.context = [latest_output]
                            
                            for branch_node in branch_nodes:
                                execution_queue.insert(0, branch_node)
                        elif isinstance(branch, (DecisionFunc, DecisionLLM)):
                            # Handle the case where a decision returns another decision node
                            # Insert at the front of the queue to be processed next
                            if verbose:
                                console.print(f"[dim]Decision returned another decision node: {escape_rich_markup(branch.description)}[/dim]")
                            execution_queue.insert(0, branch)
        
        if verbose:
            console.print("[bold green]Graph Execution Completed[/bold green]")
            spacing()
        
        return self.state
    
    def _count_all_possible_nodes(self) -> int:
        """
        Counts all possible nodes in the graph, including all branches.
        
        Returns:
            The total number of nodes in the graph
        """
        # Start with the nodes directly in the graph
        counted = set()
        to_count = list(self.nodes)
        
        while to_count:
            node = to_count.pop(0)
            if node.id in counted:
                continue
                
            counted.add(node.id)
            
            # If it's a decision node, add its branches to be counted
            if isinstance(node, (DecisionFunc, DecisionLLM)):
                if node.true_branch:
                    if isinstance(node.true_branch, TaskNode):
                        if node.true_branch.id not in counted:
                            to_count.append(node.true_branch)
                    elif isinstance(node.true_branch, TaskChain):
                        for branch_node in node.true_branch.nodes:
                            if branch_node.id not in counted:
                                to_count.append(branch_node)
                
                if node.false_branch:
                    if isinstance(node.false_branch, TaskNode):
                        if node.false_branch.id not in counted:
                            to_count.append(node.false_branch)
                    elif isinstance(node.false_branch, TaskChain):
                        for branch_node in node.false_branch.nodes:
                            if branch_node.id not in counted:
                                to_count.append(branch_node)
        
        # Return the count, minimum of 1 to avoid division by zero
        return max(len(counted), 1)
    
    def run(self, verbose: bool = True, show_progress: bool = None) -> State:
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
        return self._run_sequential(verbose, show_progress)
    
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
        
        # If not found in direct nodes, check for nodes in decision branches
        for node in self.nodes:
            if isinstance(node, (DecisionFunc, DecisionLLM)):
                # Check true branch
                if node.true_branch:
                    if isinstance(node.true_branch, TaskNode) and node.true_branch.task.description == description:
                        return self.state.get_task_output(node.true_branch.id)
                    elif isinstance(node.true_branch, TaskChain):
                        for branch_node in node.true_branch.nodes:
                            if isinstance(branch_node, TaskNode) and branch_node.task.description == description:
                                return self.state.get_task_output(branch_node.id)
                
                # Check false branch
                if node.false_branch:
                    if isinstance(node.false_branch, TaskNode) and node.false_branch.task.description == description:
                        return self.state.get_task_output(node.false_branch.id)
                    elif isinstance(node.false_branch, TaskChain):
                        for branch_node in node.false_branch.nodes:
                            if isinstance(branch_node, TaskNode) and branch_node.task.description == description:
                                return self.state.get_task_output(branch_node.id)
        
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
    elif isinstance(other, (DecisionFunc, DecisionLLM)):
        chain.add(other)
    
    return chain

# Apply the patch to the Task class
Task.__rshift__ = _task_rshift
