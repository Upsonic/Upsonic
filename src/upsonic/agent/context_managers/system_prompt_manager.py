from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Optional, List
import json

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.agent.agent import Agent
    from upsonic.tasks.tasks import Task
    from upsonic.agent.context_managers.memory_manager import MemoryManager
    from upsonic.agent.context_managers.culture_manager_context import CultureContextManager
else:
    # Use string annotations to avoid importing heavy modules
    Agent = "Agent"
    Task = "Task"
    MemoryManager = "MemoryManager"
    CultureContextManager = "CultureContextManager"


class SystemPromptManager:
    """A context manager responsible for constructing the final system prompt."""

    def __init__(self, agent: Agent, task: Task):
        """Initializes the SystemPromptManager."""
        self.agent = agent
        self.task = task
        self.system_prompt: str = ""

    def _build_system_prompt(
        self, 
        memory_handler: Optional["MemoryManager"] = None,
        culture_handler: Optional["CultureContextManager"] = None,
    ) -> str:
        """Builds the complete system prompt string by assembling its components."""
        from upsonic.tools import Thought, AnalysisResult
        from upsonic.context.agent import turn_agent_to_string
        from upsonic.context.default_prompt import default_prompt
        
        prompt_parts = []
    
        is_thinking_enabled = self.agent.enable_thinking_tool
        if self.task.enable_thinking_tool is not None:
            is_thinking_enabled = self.task.enable_thinking_tool

        is_reasoning_enabled = self.agent.enable_reasoning_tool
        if self.task.enable_reasoning_tool is not None:
            is_reasoning_enabled = self.task.enable_reasoning_tool
        
        if is_thinking_enabled:
            thought_schema_str = json.dumps(Thought.model_json_schema(), indent=2)

            if is_reasoning_enabled:
                analysis_result_schema_str = json.dumps(AnalysisResult.model_json_schema(), indent=2)

                reflective_instructions = f"""---
                ### MISSION BRIEFING: OPERATION DELIBERATE THOUGHT ###
                ---

                **YOUR ROLE: THE STRATEGIST**

                You are the strategic mind behind a complex operation. Your mission is to achieve the user's ultimate goal. You will not execute actions directly. Instead, you will formulate a high-level strategy and then delegate execution to a separate entity, the "Orchestrator."

                ---
                ### PHASE 1: STRATEGIC PLANNING (THIS IS YOUR FIRST THING TO DO)
                ---

                Your immediate and FIRST THING TO DO is to create the initial strategic plan. This plan must be delivered as a single `Thought` object, strictly conforming to the `Thought` JSON schema provided below.

                **Mandatory `Thought` Object Directives:**

                1.  **`reasoning` (string):** Articulate your strategic assessment. Deconstruct the user's request and explain the sequence of tool-based actions required. This is your "Commander's Intent."
                    *   *Example: "The user requires a conditional financial action. The logical sequence is: 1. Acquire market data. 2. Acquire trade execution capability. 3. Acquire communication capability. The Orchestrator will pause me after each action for tactical decisions."*

                2.  **`plan` (list of `PlanStep` objects):** Define the high-level sequence of tool calls.
                    *   **CRITICAL:** This plan is your "To-Do List," NOT a detailed script. Do NOT include conditional logic (if/then) or analysis steps in this initial plan. You will handle all decision-making later.
                    *   **HANDLING CONDITIONAL REQUESTS:** If the user's request contains conditional logic (e.g., "if the price is low, then buy"), your plan must still be sequential. You should first plan a step to get the data (e.g., call `get_crypto_price`), and then plan the step for the potential action (e.g., call `execute_crypto_trade`). The final decision of whether the action was *warranted* will be made by you later, during the final synthesis, based on the results.
                    *   Each `PlanStep` must contain the `tool_name` and its `parameters`.
                    *   ONLY ADD THE CORRECT TOOL NAMES AND PARAMETERS. YOU CAN USE ONLY THE TOOLS THAT ARE PROVIDED TO YOU. DO NOT ADD NON EXISTING TOOLS OR TOOLS WITH DIFFERENT NAMES, DIFFERENT PARAMETERS!
                    *   *Example plan creation: [{{'tool_name': 'get_weather', 'parameters': {{'city': 'New York'}}}}, {{'tool_name': 'get_population', 'parameters': {{'city': 'London', 'country': 'UK'}}}}, {{'tool_name': 'find_crypto_price', 'parameters': {{'crypto': 'BTC'}}}}]*
                    *   TOOL NAMES WILL BE AS DEFINED IN THE TOOLS PROVIDED TO YOU!! DO NOT ADD TOOLS LIKE THIS BELOW:
                    BAD EXAMPLE: [PlanStep(tool_name='multi_tool_use.parallel', parameters={{'tool_uses': [{{'recipient_name': 'functions.get_weather', 'parameters': {{'city': 'London'}}}}, {{'recipient_name': 'functions.get_weather', 'parameters': {{'city': 'Paris'}}}}]}}), PlanStep(tool_name='functions.analyze_weather_impact', parameters={{'weather': 'result_from_previous_step', 'activity': 'comparison'}})]
                    *   DO NOT ADD EXTRA PUNCTATIONS LIKE COMAS, NEW LINES, ETC. JUST THE TOOL NAMES AND PARAMETERS.
                    *   WE DO NOT CALL TOOLS PARALLEL, WE CALL THEM SEQUENTIALLY. SO YOUR "plan" MUST BE A SEQUENTIAL LIST OF TOOL CALLS WITH CORRECT ORDER, TOOL NAMES AND PARAMETERS.

                3.  **`criticism` (string):** Perform a pre-mission risk assessment. Identify ambiguities in the user's request or potential points of failure in your strategy.

                4.  **`action` (string: 'execute_plan' or 'request_clarification'):**
                    *   Set to `'execute_plan'` to commence the operation.
                    *   Set to `'request_clarification'` only if the mission is impossible without further intelligence from the user.

                **DO NOT** CALL ANY OTHER TOOL! JUST CALL THE `plan_and_execute` TOOL WITH YOUR COMPLETED `Thought` OBJECT.
                **FINAL ACTION FOR PHASE 1:** Call the `plan_and_execute` tool with your completed `Thought` object.

                ---
                ### PHASE 2: TACTICAL ANALYSIS (The "Act-then-Analyze" Loop)
                ---

                Once you call `plan_and_execute`, the Orchestrator takes command. It will execute one step of your plan and then **PAUSE**, re-awakening you for tactical analysis.

                **When Re-Awakened for Analysis:**

                -   **Your Context:** You will be provided with the user's original goal and the complete execution history, including the result of the last action.
                -   **Your Task:** Your sole responsibility is to provide a command to the Orchestrator by returning an `AnalysisResult` object, strictly conforming to its JSON schema.
                -   **Your `AnalysisResult` Object Directives:**
                    -   `evaluation` (string): Provide your tactical analysis of the last action's result. Is the mission proceeding as expected? Has a critical condition been met?
                    -   `next_action` (string: 'continue_plan', 'revise_plan', or 'final_answer'): This is your command to the Orchestrator.
                        -   `'continue_plan'`: The strategy is sound. Proceed to the next step in the plan.
                        -   `'revise_plan'`: The situation has changed. The current plan is invalid. You will be prompted to create a new `Thought` object.
                        -   `'final_answer'`: All necessary intelligence has been gathered. The mission can be concluded.

                ---
                ### PHASE 3: FINAL REPORT (SYNTHESIS)
                ---

                After the Orchestrator confirms the plan is complete (or you command it to conclude), you will be re-awakened a final time to deliver the mission outcome to the user. Synthesize a clean, final answer based on the full execution history. **Do not expose operational details** (reasoning, plans, analysis) in this final report.
                **DO NOT** CALL OR USE ANY OTHER TOOL! JUST RETURN THE SYNTHESIZED ANSWER AS A STRING.

                ---
                ### Note:
                ---

                You will be prompted what to do! Follow the instructions carefully.

                ---
                ###
                # SCHEMAS FOR MANDATORY COMPLIANCE
                ###
                ---

                <JSONSchemas>
                <Schema name="Thought">{thought_schema_str}</Schema>
                <Schema name="AnalysisResult">{analysis_result_schema_str}</Schema>
                </JSONSchemas>
                """
                prompt_parts.insert(0, reflective_instructions)

            else:
                reflective_instructions = f"""---
                ### MISSION BRIEFING: OPERATION BLUEPRINT ###
                ---

                **YOUR ROLE: THE ARCHITECT**

                You are a strategic architect. Your SOLE OBJECTIVE is to design a complete, static, and sequential blueprint of actions to achieve the user's goal. Once you submit this blueprint, a separate "Orchestrator" will execute it exactly as written, without any further consultation with you.

                ---
                ### PHASE 1: BLUEPRINT DESIGN (THIS IS YOUR FIRST THING TO DO)
                ---

                Your immediate and FIRST THING TO DO is to create the complete operational blueprint. This blueprint must be delivered as a single `Thought` object, strictly conforming to the `Thought` JSON schema provided below.

                **CRITICAL CONSTRAINT: This is a "fire-and-forget" mission. You will NOT be re-awakened for analysis or to change the plan. Therefore, your plan MUST be a complete, non-conditional sequence of tool calls that gathers all potentially necessary information.**

                **Mandatory `Thought` Object Directives:**

                1.  **`reasoning` (string):**
                    Provide your complete strategic analysis. Explain how your sequence of tool calls will gather all the information needed to satisfy the user's request in the final synthesis step.

                2.  **`plan` (list of `PlanStep` objects):**
                    Define the complete and final sequence of tool calls.
                    *   **HANDLING CONDITIONAL REQUESTS:** If the user's request contains conditional logic (e.g., "if the price is low, then buy"), your plan must still be sequential. You should first plan a step to get the data (e.g., call `get_crypto_price`), and then plan the step for the potential action (e.g., call `execute_crypto_trade`). The final decision of whether the action was *warranted* will be made by you later, during the final synthesis, based on the results.
                    *   Each `PlanStep` must contain the `tool_name` and its `parameters`.
                    *   ONLY ADD THE CORRECT TOOL NAMES AND PARAMETERS. YOU CAN USE ONLY THE TOOLS THAT ARE PROVIDED TO YOU. DO NOT ADD NON EXISTING TOOLS OR TOOLS WITH DIFFERENT NAMES, DIFFERENT PARAMETERS!
                    *   *Example plan creation: [{{'tool_name': 'get_weather', 'parameters': {{'city': 'New York'}}}}, {{'tool_name': 'get_population', 'parameters': {{'city': 'London', 'country': 'UK'}}}}, {{'tool_name': 'find_crypto_price', 'parameters': {{'crypto': 'BTC'}}}}]*
                    *   TOOL NAMES WILL BE AS DEFINED IN THE TOOLS PROVIDED TO YOU!! DO NOT ADD TOOLS LIKE THIS BELOW:
                    BAD EXAMPLE: [PlanStep(tool_name='multi_tool_use.parallel', parameters={{'tool_uses': [{{'recipient_name': 'functions.get_weather', 'parameters': {{'city': 'London'}}}}, {{'recipient_name': 'functions.get_weather', 'parameters': {{'city': 'Paris'}}}}]}}), PlanStep(tool_name='functions.analyze_weather_impact', parameters={{'weather': 'result_from_previous_step', 'activity': 'comparison'}})]
                    *   DO NOT ADD EXTRA PUNCTATIONS LIKE COMAS, NEW LINES, ETC. JUST THE TOOL NAMES AND PARAMETERS.

                3.  **`criticism` (string):**
                    Perform a risk assessment of your blueprint. Are there any ambiguities? Does the sequential plan account for all parts of the user's request?

                4.  **`action` (string: 'execute_plan' or 'request_clarification'):**
                    *   Set to `'execute_plan'` to submit your final blueprint for execution.
                    *   Set to `'request_clarification'` only if the blueprint cannot be designed without more information from the user.
                **DO NOT** CALL ANY OTHER TOOL! JUST CALL THE `plan_and_execute` TOOL WITH YOUR COMPLETED `Thought` OBJECT.
                **FINAL ACTION:** Call the `plan_and_execute` tool with your completed `Thought` object.

                ---
                ### PHASE 2: AUTOMATED EXECUTION (The Orchestrator's Role)
                ---

                Once you call `plan_and_execute`, your job is done until the final report. The Orchestrator will take your blueprint and execute **every step sequentially and non-interactively**.

                ---
                ### PHASE 3: FINAL SYNTHESIS (Your Final Report)
                ---

                After the Orchestrator has completed all steps, you will be re-awakened one last time. You will be given the full, unchangeable history of the execution. Your final task is to:
                1.  Analyze the results.
                2.  Apply any conditional logic from the original request IF YOU HAVE TO, OTHERWEISE DO NOT CALL ANY TOOL IF ALL THE INFORMATION REQUIRED IS PRESENT. (e.g., "now I see the price was below the threshold, so the trade I planned was correct").
                3.  Synthesize a final, comprehensive answer for the user.

                ---
                ### Note:
                ---

                You will be prompted what to do! Follow the instructions carefully.

                ---
                ###
                # SCHEMA FOR MANDATORY COMPLIANCE
                ###
                ---

                <JSONSchemas>
                <Schema name="Thought">{thought_schema_str}</Schema>
                </JSONSchemas>
                """
            prompt_parts.insert(0, reflective_instructions)

        base_prompt = ""

        if memory_handler:
            system_injection = memory_handler.get_system_prompt_injection()
            if system_injection:
                prompt_parts.append(system_injection)

        # Inject cultural knowledge context if available
        if culture_handler:
            culture_context = culture_handler.get_culture_context()
            if culture_context:
                prompt_parts.append(culture_context)

        if self.agent.system_prompt:
            base_prompt = self.agent.system_prompt
        
        has_any_info = False

        if self.agent.role:
            base_prompt += f"\nThis is your role: {self.agent.role}"
            has_any_info = True
        if self.agent.goal:
            base_prompt += f"\nThis is your goal: {self.agent.goal}"
            has_any_info = True
        if self.agent.instructions:
            base_prompt += f"\nThis is your instructions to follow: {self.agent.instructions}"
            has_any_info = True
        if self.agent.education:
            base_prompt += f"\nThis is your education: {self.agent.education}"
            has_any_info = True
        if self.agent.work_experience:
            base_prompt += f"\nThis is your work experiences: {self.agent.work_experience}"
            has_any_info = True

        if self.agent.company_name:
            base_prompt += f"\nYour company name is: {self.agent.company_name}"
            has_any_info = True
        if self.agent.company_url:
            base_prompt += f"\nYour company website is: {self.agent.company_url}"
            has_any_info = True
        if self.agent.company_objective:
            base_prompt += f"\nYour company objective is: {self.agent.company_objective}"
            has_any_info = True
        if self.agent.company_description:
            base_prompt += f"\nYour company description is: {self.agent.company_description}"
            has_any_info = True
            
        
        
        if not self.agent.system_prompt and not has_any_info and not is_thinking_enabled:
            base_prompt = default_prompt().prompt
        
        prompt_parts.append(base_prompt.strip())

        agent_context_str = "<YourCharacter>"
        found_agent_context = False

        if self.task.context:
            for item in self.task.context:
                if isinstance(item, type(self.agent)):
                    agent_context_str += f"\nAgent ID ({item.get_agent_id()}): {turn_agent_to_string(item)}"
                    found_agent_context = True
        
        if found_agent_context:
            agent_context_str += "\n</YourCharacter>"
            prompt_parts.append(agent_context_str)
        return "\n\n".join(prompt_parts)
    
    def get_system_prompt(self) -> str:
        """
        Public getter to retrieve the constructed system prompt.

        This method is called from within the `do_async` pipeline after this
        manager has been entered.

        Returns:
            The final system prompt string.
        """
        return self.system_prompt
    
    def should_include_system_prompt(self, message_history: List[Any]) -> bool:
        """
        Determines whether the system prompt should be included in the current request.
        
        The system prompt should be included if:
        1. There are no messages in history (first request), OR
        2. The first message doesn't have a system prompt, OR
        3. The system prompt contains dynamic content (like user profile) that should always be included
        
        Args:
            message_history: List of previous messages in the conversation
            
        Returns:
            True if system prompt should be included, False otherwise
        """
        from upsonic.messages import SystemPromptPart, ModelRequest
        
        # If no messages, always include system prompt
        if not message_history:
            return True
        
        # Check if first message has a system prompt
        has_system_prompt_in_history = False
        if message_history and isinstance(message_history[0], ModelRequest):
            has_system_prompt_in_history = any(
                isinstance(part, SystemPromptPart) 
                for part in message_history[0].parts
            )
        
        # If first message doesn't have system prompt, we should add it
        if not has_system_prompt_in_history:
            return True
        
        # If system prompt contains dynamic content (user profile, culture, etc.), 
        # always include it to ensure latest information is available
        system_prompt = self.get_system_prompt()
        if system_prompt:
            # Check if system prompt contains dynamic content markers
            has_user_profile = "<UserProfile>" in system_prompt
            has_culture_context = "<CulturalKnowledge>" in system_prompt or "<CultureContext>" in system_prompt
            
            # Always include if it has dynamic content that may have changed
            if has_user_profile or has_culture_context:
                return True
        
        # Default: don't include if first message already has system prompt
        # (to avoid duplicate system prompts)
        return False

    async def aprepare(
        self, 
        memory_handler: Optional["MemoryManager"] = None,
        culture_handler: Optional["CultureContextManager"] = None,
    ) -> None:
        """
        Prepare the system prompt before the LLM call.
        
        Args:
            memory_handler: Optional MemoryManager for memory injection
            culture_handler: Optional CultureContextManager for culture injection
        """
        self.system_prompt = self._build_system_prompt(memory_handler, culture_handler)
    
    async def afinalize(self) -> None:
        """Finalize system prompt after the LLM call."""
        pass
    
    def prepare(
        self, 
        memory_handler: Optional["MemoryManager"] = None,
        culture_handler: Optional["CultureContextManager"] = None,
    ) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare(memory_handler, culture_handler))
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())

    @asynccontextmanager
    async def manage_system_prompt(
        self, 
        memory_handler: Optional["MemoryManager"] = None,
        culture_handler: Optional["CultureContextManager"] = None,
    ):
        """
        The asynchronous context manager for building the system prompt.

        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        
        Args:
            memory_handler: Optional MemoryManager for memory injection
            culture_handler: Optional CultureContextManager for culture injection
        """
        await self.aprepare(memory_handler, culture_handler)
            
        try:
            yield self
        finally:
            await self.afinalize()