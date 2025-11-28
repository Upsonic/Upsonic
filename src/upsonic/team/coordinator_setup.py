from __future__ import annotations
import inspect
from typing import TYPE_CHECKING, List, Any, Callable, Literal

if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.tasks.tasks import Task
    from upsonic.knowledge_base.knowledge_base import KnowledgeBase

class CoordinatorSetup:
    """
    Manages the setup and configuration of the Team Leader agent.
    
    This class is now mode-aware and can generate different system prompts
    for different team operational modes ('coordinate' or 'route').
    """
    def __init__(self, members: List[Agent], tasks: List[Task], mode: Literal["coordinate", "route"]):
        """
        Initializes the CoordinatorSetup manager.

        Args:
            members (List[Agent]): The list of member agents available to the team.
            tasks (List[Task]): The initial list of tasks for the team to accomplish.
            mode (Literal["coordinate", "route"]): The operational mode for the team.
        """
        self.members = members
        self.tasks = tasks
        self.mode = mode

    def _summarize_tool(self, tool: Callable) -> str:
        """
        Creates a human-readable summary of a tool from its name and docstring.
        """
        tool_name = getattr(tool, '__name__', 'Unnamed Tool')
        docstring = inspect.getdoc(tool)
        if docstring:
            description = docstring
        else:
            description = "No description available."
        return f"{tool_name}: {description}"

    def _format_agent_manifest(self) -> str:
        if not self.members:
            return "No team members are available."
        manifest_parts = []
        for agent in self.members:
            agent_id = agent.get_agent_id()
            role = agent.role or "No specific role defined."
            goal = agent.goal or "No specific goal defined."
            system_prompt = agent.system_prompt or "No system prompt defined."
            
            # Include agent tools if available
            tools_info = ""
            if hasattr(agent, 'tools') and agent.tools:
                tool_summaries = [self._summarize_tool(tool) for tool in agent.tools]
                tools_str = "\n    ".join([f"- {summary}" for summary in tool_summaries])
                tools_info = f"\n  - Agent Tools:\n    {tools_str}"
            
            part = f"- Member ID: `{agent_id}`\n  - Role: {role}\n  - Goal: {goal}\n  - System Prompt: {system_prompt}\n  -Agent Tools: {tools_info}"
            manifest_parts.append(part)
        return "\n".join(manifest_parts)

    def _serialize_context_item(self, item: Any) -> str:
        from upsonic.tasks.tasks import Task
        from upsonic.knowledge_base.knowledge_base import KnowledgeBase
        if isinstance(item, str):
            return item
        if isinstance(item, Task):
            return f"Reference to another task with description: '{item.description}'"
        if isinstance(item, KnowledgeBase):
            return f"Reference to KnowledgeBase '{item.name}' containing markdown or RAG-enabled content."
        try:
            return str(item)
        except Exception:
            return "Unserializable context object."

    def _format_tasks_manifest(self) -> str:
        if not self.tasks:
            return "<Tasks>\nNo initial tasks provided.\n</Tasks>"

        manifest_parts = ["<Tasks>"]
        for i, task in enumerate(self.tasks, 1):
            task_parts = [f"  <Task index='{i}'>"]
            task_parts.append(f"    <Description>{task.description}</Description>")

            if task.tools:
                summaries = [self._summarize_tool(tool) for tool in task.tools]
                tools_str = "\n".join([f"      - {summary}" for summary in summaries])
                task_parts.append(f"    <Tools>\n{tools_str}\n    </Tools>")
            else:
                task_parts.append("    <Tools>None</Tools>")

            if task.context:
                context_items = [self._serialize_context_item(item) for item in task.context]
                context_str = "\n".join([f"      - {item}" for item in context_items])
                task_parts.append(f"    <Context>\n{context_str}\n    </Context>")
            else:
                 task_parts.append("    <Context>None</Context>")

            if task.attachments:
                attachment_str = ", ".join(task.attachments)
                task_parts.append(f"    <Attachments>{attachment_str}</Attachments>")
            else:
                task_parts.append("    <Attachments>None</Attachments>")

            task_parts.append("  </Task>")
            manifest_parts.append("\n".join(task_parts))
        
        manifest_parts.append("</Tasks>")
        return "\n".join(manifest_parts)
    
    def create_leader_prompt(self) -> str:
        """
        Constructs the complete system prompt for the Team Leader agent
        based on the team's operational mode.
        """
        if self.mode == "coordinate":
            return self._create_coordinate_prompt()
        elif self.mode == "route":
            return self._create_route_prompt()
        else:
            # Fallback for safety
            return "You are a helpful assistant."    

    def _create_coordinate_prompt(self) -> str:
        """
        Constructs the complete system prompt for the Team Leader agent,
        including manifests for both team members and initial tasks with full tool schemas.
        """
        members_manifest = self._format_agent_manifest()
        tasks_manifest = self._format_tasks_manifest()

        leader_system_prompt = (
            "### IDENTITY AND MISSION ###\n"
            "You are the Strategic Coordinator of an elite team of specialized AI agents. Your SOLE function is to achieve the user's objectives by orchestrating your team. You do not perform tasks yourself; you analyze, plan, delegate, and synthesize.\n\n"
            "--- INTEL-PACKAGE ---\n"
            "This is the complete intelligence available for your mission.\n\n"

            "**1. TEAM ROSTER:**\n"
            f"{members_manifest}\n\n"
            
            "**2. MISSION OBJECTIVES (INITIAL TASKS):**\n"
            f"{tasks_manifest}\n\n"

            "--- OPERATIONAL PROTOCOL ---\n"
            "You must adhere to the following protocol for mission execution:\n\n"

            "**1. Analyze:** Review all `<Task>` blocks in your MISSION OBJECTIVES. Note the descriptions and required tool names. Formulate a step-by-step plan to achieve the objectives, deciding which member is best suited for each step.\n\n"
            
            "**2. Delegate:** To assign a sub-task, you MUST call your one and only tool, `delegate_task`. This tool accepts several parameters to precisely define the sub-task.\n\n"

            "   **`delegate_task` Parameters:**\n"
            "   - `member_id` (string, **required**): The ID of the agent you are assigning the task to.\n"
            "   - `description` (string, **required**): A clear, self-contained description of what the member needs to do.\n"
            "   - `tools` (List[string], optional): A list of tool **names** that the member may need to use. You should derive these from the `<Tools>` tag in the initial objectives.\n"
            "   - `context` (Any, optional): The result from a previous step or any other data the member needs to complete their task.\n"
            "   - `attachments` (List[string], optional): A list of file paths the member needs.\n\n"

            "   **CRITICAL EXAMPLE - HOW TO DELEGATE:**\n"
            "   *   **Your Thought Process:** 'The first task requires the `get_crypto_price` tool. The `Crypto_Data_Fetcher` is the expert for this. I will call `delegate_task`.'\n"
            "   *   **Resulting Tool Call:** Your agent's internal reasoning would generate a call equivalent to this:\n"
            "     `delegate_task(\n"
            "       member_id='Crypto_Data_Fetcher',\n"
            "       description='Find the current price of Ethereum (ETH) and return it as a JSON string.',\n"
            "       tools=['get_crypto_price']\n"
            "     )`\n\n"

            "**3. Synthesize:** After a member returns a result, use it as `context` for the next step if necessary. Once all objectives are met, combine all results into a single, comprehensive final answer. Do not mention your internal processes in the final report."
        )
        return leader_system_prompt
    
    def _create_route_prompt(self) -> str:
        """Generates the new, specialized system prompt for the 'route' mode."""
        members_manifest = self._format_agent_manifest()
        tasks_manifest = self._format_tasks_manifest()

        return (
            "### IDENTITY AND MISSION ###\n"
            "You are an intelligent AI Router. Your SOLE purpose is to analyze the user's full request and determine which single specialist agent on your team is best suited to handle the entire set of objectives. You do not answer the query yourself; you only decide who should.\n\n"
            
            "--- INTEL-PACKAGE ---\n"
            "**1. TEAM ROSTER:** This is the list of available specialists.\n"
            f"{members_manifest}\n\n"
            
            "**2. MISSION OBJECTIVES (INITIAL TASKS):** This is the complete user request you must route.\n"
            f"{tasks_manifest}\n\n"

            "--- OPERATIONAL PROTOCOL ---\n"
            "Your process is a strict two-step sequence:\n\n"
            
            "**1. Analyze and Decide:**\n"
            "   - Read all `<Task>` blocks in the MISSION OBJECTIVES. Pay close attention to the `<Description>` and the required capabilities listed in the `<Tools>` tag for each task.\n"
            "   - Compare the overall requirements of the mission against the `role` and `goal` of each agent in your TEAM ROSTER.\n"
            "   - Select the **single best agent** whose skills most closely match the entire set of tasks.\n\n"

            "**2. Execute Handoff:**\n"
            "   - Once you have made your final decision, you MUST call your one and only tool, `route_request_to_member`.\n"
            "   - Provide the `member_id` of your chosen agent as the sole argument.\n"
            "   - This is your final action. Your job is complete after making this tool call.\n\n"

            "### FINAL DIRECTIVE ###\n"
            "Do not attempt to answer the user's query or break it down. Your only task is to analyze the full mission objective and route it to the single most qualified specialist. Make one tool call and then stop."
        )