from typing import Any, List, Tuple, Set, Dict, Optional
import io

from upsonic.lcel.runnable import Runnable


class RunnableNode:
    """A node in the runnable graph."""
    
    def __init__(self, runnable: Runnable, node_id: int):
        self.runnable = runnable
        self.node_id = node_id
        self.name = self._get_name()
        self.edges_to: List[int] = []  # Sequential edges (to next step)
        self.parallel_branches: List[Tuple[str, int]] = []  # (key_name, node_id) for parallel branches
    
    def _get_name(self) -> str:
        """Get a human-readable name for this runnable."""
        # Use class name and simplify common patterns
        class_name = self.runnable.__class__.__name__
        
        # Special handling for common types
        if hasattr(self.runnable, '__repr__'):
            repr_str = repr(self.runnable)
            
            # Remove object memory addresses
            import re
            repr_str = re.sub(r'<.*? object at 0x[0-9a-f]+>', class_name, repr_str)
            
            # Limit length for display
            if len(repr_str) < 50:
                return repr_str
        
        return class_name


class RunnableGraph:
    """Graph representation of a runnable chain."""
    
    def __init__(self, root: Runnable):
        """
        Initialize the graph from a root runnable.
        
        Args:
            root: The root runnable to build the graph from
        """
        self.root = root
        self.nodes: Dict[int, RunnableNode] = {}
        self.node_counter = 0
        self._visited: Set[int] = set()
        
        # Build the graph
        self._build_graph(root)
    
    def _build_graph(self, runnable: Runnable, parent_id: Optional[int] = None) -> int:
        """
        Build the graph structure showing main chain components.
        
        This expands RunnableSequence to show sequential flow and
        RunnableParallel to show parallel branches as internal structure.
        
        Args:
            runnable: The runnable to process
            parent_id: ID of the parent node
            
        Returns:
            The node ID of the processed runnable (entry point)
        """
        from upsonic.lcel.sequence import RunnableSequence
        from upsonic.lcel.parallel import RunnableParallel
        
        # If it's a RunnableSequence, expand it to show the chain
        if isinstance(runnable, RunnableSequence):
            # Don't create a node for RunnableSequence itself,
            # just create nodes for its steps and connect them sequentially
            prev_id = None
            first_step_id = None
            
            for step in runnable.steps:
                step_id = self._build_graph(step)  # Recursive to handle nested structures
                
                if first_step_id is None:
                    first_step_id = step_id
                
                if prev_id is not None:
                    # Connect previous step to current step (sequential edge)
                    self.nodes[prev_id].edges_to.append(step_id)
                
                prev_id = step_id
            
            return first_step_id if first_step_id is not None else self.node_counter
        
        # If it's a RunnableParallel, create a node and show its branches as internal
        elif isinstance(runnable, RunnableParallel):
            # Create a node for the RunnableParallel itself
            parallel_node_id = self._build_graph_node(runnable)
            
            # Create nodes for each parallel branch with their key names
            for name, step in runnable.steps.items():
                step_id = self._build_graph(step)  # Recursive to handle nested structures
                # Add as parallel branch with key name (internal structure)
                self.nodes[parallel_node_id].parallel_branches.append((name, step_id))
            
            return parallel_node_id
        
        else:
            # For all other runnables, create a single node
            return self._build_graph_node(runnable)
    
    def _build_graph_node(self, runnable: Runnable) -> int:
        """
        Create a single node for a runnable without expanding its internal structure.
        
        Args:
            runnable: The runnable to create a node for
            
        Returns:
            The node ID
        """
        node_id = self.node_counter
        self.node_counter += 1
        
        node = RunnableNode(runnable, node_id)
        self.nodes[node_id] = node
        
        return node_id
    
    def print_ascii(self) -> None:
        """Print an ASCII representation of the graph."""
        print(self.to_ascii())
    
    def to_ascii(self) -> str:
        """Generate an ASCII representation of the graph."""
        if not self.nodes:
            return "Empty graph"
        
        # Build a representation that handles both sequential and parallel structures
        lines = []
        
        # Find the start node (node with no incoming edges)
        incoming_edges = set()
        for node in self.nodes.values():
            incoming_edges.update(node.edges_to)
        
        start_nodes = [node_id for node_id in self.nodes.keys() if node_id not in incoming_edges]
        
        if not start_nodes:
            # If no clear start (circular reference), start from node 0
            start_nodes = [0]
        
        # Traverse the graph from start node
        visited = set()
        
        def traverse(node_id: int, depth: int = 0):
            if node_id in visited or node_id not in self.nodes:
                return
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            # Add node to output
            indent = "  " * depth
            lines.append(f"{indent}{node.name}")
            
            # Show parallel branches (internal structure)
            if node.parallel_branches:
                for i, (key_name, branch_id) in enumerate(node.parallel_branches):
                    is_last = i == len(node.parallel_branches) - 1
                    branch_node = self.nodes.get(branch_id)
                    if branch_node:
                        # Mark branch as visited so it doesn't appear as a separate top-level node
                        visited.add(branch_id)
                        
                        branch_indent = indent + "  "
                        if is_last:
                            lines.append(f"{branch_indent}└─> [{key_name}] {branch_node.name}")
                        else:
                            lines.append(f"{branch_indent}├─> [{key_name}] {branch_node.name}")
                        
                        # Recursively show branches of branches (if they have parallel branches)
                        if branch_node.parallel_branches:
                            for sub_key, sub_branch_id in branch_node.parallel_branches:
                                if sub_branch_id not in visited:
                                    traverse(sub_branch_id, depth + 2)
            
            # Show sequential connection to next step
            if node.edges_to:
                lines.append(f"{indent}  |")
                lines.append(f"{indent}  v")
                for child_id in node.edges_to:
                    traverse(child_id, depth)
        
        # Start traversal
        for start_node in start_nodes:
            traverse(start_node)
        
        return "\n".join(lines)
    
    def to_mermaid(self) -> str:
        """Generate a Mermaid diagram representation of the graph."""
        lines = ["graph TD"]  # Top to bottom for better readability
        
        # Add nodes with cleaned labels
        for node_id, node in self.nodes.items():
            # Clean up the label for better display
            label = node.name
            # Remove object memory addresses
            import re
            label = re.sub(r'<.*? object at 0x[0-9a-f]+>', '', label)
            # Escape special characters for Mermaid
            label = label.replace('"', "'")
            # Limit label length
            if len(label) > 40:
                label = label[:37] + "..."
            
            lines.append(f'    {node_id}["{label}"]')
        
        # Add edges - show parallel execution and merge
        for node_id, node in self.nodes.items():
            # If this node has parallel branches AND sequential edges (next step)
            if node.parallel_branches and node.edges_to:
                # Show parallel split with key names
                for key_name, branch_id in node.parallel_branches:
                    lines.append(f'    {node_id} -.->|{key_name}| {branch_id}')
                
                # Show merge: all parallel branches connect to next node
                next_node_id = node.edges_to[0]
                for key_name, branch_id in node.parallel_branches:
                    lines.append(f'    {branch_id} --> {next_node_id}')
                
                # Show continuation (but don't duplicate the connection)
                # The merge already shows connection to next_node_id
                for i, target_id in enumerate(node.edges_to):
                    if i > 0:  # Skip first one as it's already shown via merge
                        lines.append(f'    {target_id} ==> {node.edges_to[i]}')
            
            # If no parallel branches, just show sequential edges
            elif node.edges_to:
                for target_id in node.edges_to:
                    lines.append(f'    {node_id} ==> {target_id}')
            
            # If only parallel branches (no next step), just show the branches
            elif node.parallel_branches:
                for key_name, branch_id in node.parallel_branches:
                    lines.append(f'    {node_id} -.->|{key_name}| {branch_id}')
        
        return "\n".join(lines)
    
    def get_structure_details(self) -> str:
        """Get detailed structure information about the graph."""
        lines = []
        
        # First pass: build a map of which nodes are merge targets
        merge_targets = {}  # branch_node_id -> merge_target_node_id
        for node_id, node in self.nodes.items():
            if node.parallel_branches and node.edges_to:
                # These branches merge to the next sequential node
                next_node = node.edges_to[0]
                for key, branch_id in node.parallel_branches:
                    merge_targets[branch_id] = next_node
        
        # Second pass: display all nodes with merge information
        for node_id, node in sorted(self.nodes.items()):
            lines.append(f"Node {node_id}: {node.name}")
            
            if node.parallel_branches:
                branch_info = [f"'{key}' -> Node {bid}" for key, bid in node.parallel_branches]
                lines.append(f"  Parallel branches: {', '.join(branch_info)}")
                
                # Show merge connections if there's a next step
                if node.edges_to:
                    next_node = node.edges_to[0]
                    merge_info = [f"Node {bid}" for _, bid in node.parallel_branches]
                    lines.append(f"  Merge: {', '.join(merge_info)} --> Node {next_node}")
            
            if node.edges_to:
                lines.append(f"  Sequential edges to: {node.edges_to}")
            
            # Check if this node is a parallel branch that merges somewhere
            if node_id in merge_targets:
                merge_to = merge_targets[node_id]
                lines.append(f"  Merges to: Node {merge_to}")
            elif not node.parallel_branches and not node.edges_to:
                lines.append(f"  (leaf node)")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        num_nodes = len(self.nodes)
        num_sequential_edges = sum(len(node.edges_to) for node in self.nodes.values())
        num_parallel_branches = sum(len(node.parallel_branches) for node in self.nodes.values())
        
        # Count merge edges: each parallel branch merges to the next node
        num_merge_edges = 0
        for node in self.nodes.values():
            if node.parallel_branches and node.edges_to:
                # Each branch merges to the next node
                num_merge_edges += len(node.parallel_branches)
        
        # Total edges = sequential + parallel splits + merges
        total_edges = num_sequential_edges + num_parallel_branches + num_merge_edges
        
        return f"RunnableGraph(nodes={num_nodes}, edges={total_edges})"
