"""
CultureManager for handling agent culture and behavior guidelines.

This manager handles:
- Accepting user-provided Culture instances
- Using an Agent to extract structured guidelines from descriptions
- Formatting culture guidelines for system prompt injection
- Managing repeat intervals for culture injection

Storage operations are NOT handled by CultureManager (removed from storage).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from upsonic.culture.culture import Culture
    from upsonic.models import Model

# System prompt for the culture extraction agent
CULTURE_EXTRACTION_SYSTEM_PROMPT = """You are a Culture Extraction Agent responsible for analyzing user descriptions and extracting structured behavioral guidelines.

Your role is to:
1. Analyze user-provided descriptions of desired agent behavior
2. Extract four key aspects of agent culture:
   - Tone of Speech: How the agent should communicate (formal, friendly, professional, etc.)
   - Topics I Shouldn't Talk About: Subjects the agent should avoid or decline
   - Topics I Can Help With: Subjects the agent is knowledgeable and helpful about
   - Things I Should Pay Attention To: Important considerations, principles, or guidelines

Guidelines for extraction:
- Be specific and actionable
- Extract concrete guidelines, not vague suggestions
- Preserve the intent and context of the original description
- Format each section clearly and concisely
- If a section doesn't apply, use "N/A" or leave it minimal
"""


class CultureManager:
    """Manager for agent culture and behavior guidelines.
    
    CultureManager handles:
    1. Accepting user-provided Culture instances
    2. Using an Agent to extract structured guidelines from descriptions
    3. Formatting culture guidelines for system prompt injection
    4. Managing repeat intervals for periodic culture injection
        
    Usage:
        from upsonic.culture import Culture
        
        culture = Culture(
            description="You are a 5-star hotel receptionist",
            add_system_prompt=True,
            repeat=False,
            repeat_interval=5
        )
        
        manager = CultureManager(model="openai/gpt-4o")
        manager.set_culture(culture)
        await manager.aprepare()  # This processes the description
        formatted = manager.format_for_system_prompt()
    """
    
    def __init__(
        self,
        model: Optional[Union["Model", str]] = None,
        enabled: bool = True,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        debug: bool = False,
        debug_level: int = 1,
        print: Optional[bool] = None,
    ) -> None:
        """
        Initialize the CultureManager.
        
        Args:
            model: Model for culture extraction
            enabled: Whether culture management is enabled
            agent_id: Agent ID for culture context
            team_id: Team ID for culture context
            debug: Enable debug logging
            debug_level: Debug verbosity level (1-3)
            print: Enable printing of output. If None, uses default (True). Should match main agent's print flag.
        """
        self._model_spec = model
        self.enabled = enabled
        self.agent_id = agent_id
        self.team_id = team_id
        self.debug = debug
        self.debug_level = debug_level
        self.print = print if print is not None else True
        
        # Current culture instance
        self._culture: Optional["Culture"] = None
        
        # Extracted culture guidelines (processed from description)
        self._extracted_guidelines: Optional[Dict[str, str]] = None
        
        # Track if culture was prepared
        self._prepared: bool = False
        
        # Track message count for repeat logic
        self._message_count: int = 0
        
        # Accumulated LLM usage from sub-agent calls (culture extraction)
    
    @property
    def culture(self) -> Optional["Culture"]:
        """Get the current culture instance."""
        return self._culture
    
    @property
    def extracted_guidelines(self) -> Optional[Dict[str, str]]:
        """Get the extracted culture guidelines."""
        return self._extracted_guidelines
    
    @property
    def prepared(self) -> bool:
        """Check if culture has been prepared."""
        return self._prepared
    
    def set_culture(self, culture: "Culture") -> None:
        """
        Set culture from user input.
        
        Args:
            culture: Culture instance with description and settings
        """
        self._culture = culture
        self._prepared = False
        self._extracted_guidelines = None
    
    async def aprepare(self) -> None:
        """
        Prepare culture by extracting guidelines from description.
        
        This method should be called after set_culture() to process
        the description and extract structured guidelines.
        """
        if self._prepared or not self._culture:
            return
        
        self._prepared = True
        
        # Extract guidelines from description
        await self._extract_guidelines(self._culture.description)
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        try:
            _ = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, self.aprepare()).result()
        except RuntimeError:
            asyncio.run(self.aprepare())
    
    async def _extract_guidelines(self, description: str) -> Dict[str, str]:
        """
        Extract structured guidelines from culture description using an Agent.
        
        Args:
            description: User description of desired agent behavior
            
        Returns:
            Dictionary with extracted guidelines:
            - tone_of_speech
            - topics_to_avoid
            - topics_to_help
            - things_to_pay_attention
        """
        from pydantic import BaseModel, Field
        from upsonic.utils.printing import info_log, warning_log
        
        if not self._model_spec:
            if self.debug:
                warning_log(
                    "CultureManager: No model configured, using basic extraction",
                    "CultureManager"
                )
            # Basic fallback
            self._extracted_guidelines = {
                "tone_of_speech": "Professional and helpful",
                "topics_to_avoid": "N/A",
                "topics_to_help": "General assistance",
                "things_to_pay_attention": "User needs and preferences"
            }
            return self._extracted_guidelines
        
        # Define output schema for extraction
        class ExtractedCulture(BaseModel):
            """Extracted culture guidelines from user description."""
            tone_of_speech: str = Field(
                ...,
                description="How the agent should communicate (formal, friendly, professional, etc.)"
            )
            topics_to_avoid: str = Field(
                ...,
                description="Subjects the agent should avoid or decline to discuss"
            )
            topics_to_help: str = Field(
                ...,
                description="Subjects the agent is knowledgeable and helpful about"
            )
            things_to_pay_attention: str = Field(
                ...,
                description="Important considerations, principles, or guidelines the agent should follow"
            )
        
        # Create extraction task
        extraction_prompt = f"""Analyze the following user description and extract structured culture guidelines.

User Description:
{description}

Extract the four key aspects:
1. Tone of Speech: How should the agent communicate?
2. Topics I Shouldn't Talk About: What should the agent avoid?
3. Topics I Can Help With: What is the agent knowledgeable about?
4. Things I Should Pay Attention To: What principles should guide the agent?

Provide specific, actionable guidelines for each aspect.
"""
        
        try:
            from upsonic.agent.agent import Agent
            from upsonic.tasks.tasks import Task
            
            extractor = Agent(
                model=self._model_spec,
                name="Culture Extractor",
                system_prompt=CULTURE_EXTRACTION_SYSTEM_PROMPT,
                debug=self.debug,
                debug_level=self.debug_level,
                print=self.print,
            )
            
            task = Task(
                description=extraction_prompt,
                response_format=ExtractedCulture,
            )
            
            extractor_output = await extractor.do_async(task, return_output=True)
            result = extractor_output.output

            # The extractor sub-agent's LLM call already lands in the
            # usage registry under the parent's inherited scope tags,
            # so no manual aggregation is needed here.

            if result:
                self._extracted_guidelines = {
                    "tone_of_speech": result.tone_of_speech if result and hasattr(result, 'tone_of_speech') else "Professional and helpful",
                    "topics_to_avoid": result.topics_to_avoid if result and hasattr(result, 'topics_to_avoid') else "N/A",
                    "topics_to_help": result.topics_to_help if result and hasattr(result, 'topics_to_help') else "General assistance",
                    "things_to_pay_attention": result.things_to_pay_attention if result and hasattr(result, 'things_to_pay_attention') else "User needs and preferences",
                }
                
                if self.debug:
                    info_log(
                        "Extracted culture guidelines from description",
                        "CultureManager"
                    )
                
                return self._extracted_guidelines
            else:
                if self.debug:
                    warning_log("Culture extraction returned no result, using basic fallback", "CultureManager")
                # Basic fallback
                self._extracted_guidelines = {
                    "tone_of_speech": "Professional and helpful",
                    "topics_to_avoid": "N/A",
                    "topics_to_help": "General assistance",
                    "things_to_pay_attention": "User needs and preferences"
                }
                return self._extracted_guidelines
                
        except Exception as e:
            if self.debug:
                warning_log(f"Culture extraction failed: {e}, using basic fallback", "CultureManager")
            # Basic fallback
            self._extracted_guidelines = {
                "tone_of_speech": "Professional and helpful",
                "topics_to_avoid": "N/A",
                "topics_to_help": "General assistance",
                "things_to_pay_attention": "User needs and preferences"
            }
            return self._extracted_guidelines
    
    def format_for_system_prompt(self) -> Optional[str]:
        """
        Format culture guidelines for system prompt injection.
        
        Returns:
            Formatted string for system prompt wrapped in <CulturalKnowledge> tags, or None if no culture
        """
        if not self._culture or not self._extracted_guidelines:
            return None
        
        guidelines = self._extracted_guidelines
        
        parts = []
        parts.append("## MANDATORY AGENT CULTURE GUIDELINES - STRICT COMPLIANCE REQUIRED")
        parts.append("")
        parts.append("**CRITICAL: You MUST strictly adhere to these culture guidelines. These guidelines define your identity, behavior, and scope of assistance. You MUST NOT deviate from these guidelines under any circumstances.**")
        parts.append("")
        parts.append("### Your Identity")
        parts.append(f"**{self._culture.description}**")
        parts.append("")
        parts.append("You MUST embody this identity completely. Any specific names, roles, organizations, or locations mentioned above are YOUR identity - memorize and use them accurately when asked.")
        parts.append("")
        parts.append("**IDENTITY AND CHARACTER:**")
        parts.append("You MUST respond and behave EXACTLY as described in these culture guidelines. You are NOT a general-purpose AI assistant. You have a specific role, identity, and scope defined by these guidelines.")
        parts.append("")
        parts.append("**SCOPE ENFORCEMENT:**")
        parts.append("- You MUST ONLY help with topics listed in 'Topics I Can Help With'")
        parts.append("- You MUST decline and redirect ANY topic NOT listed in 'Topics I Can Help With'")
        parts.append("- You MUST NOT discuss topics listed in 'Topics I Shouldn't Talk About'")
        parts.append("- When asked about topics outside your scope, politely decline and redirect to your defined scope")
        parts.append("- You MUST stay in character at all times - do not break character or acknowledge you are an AI unless explicitly required by your role")
        parts.append("")
        parts.append("### Tone of Speech")
        parts.append(guidelines.get("tone_of_speech", "N/A"))
        parts.append("")
        
        parts.append("### Topics I Shouldn't Talk About")
        topics_to_avoid = guidelines.get("topics_to_avoid", "N/A")
        parts.append(topics_to_avoid)
        if topics_to_avoid and topics_to_avoid != "N/A":
            parts.append("")
            parts.append("**IMPORTANT:** If a user asks about any of these topics, you MUST politely decline and explain that these topics are outside your scope.")
        parts.append("")
        
        parts.append("### Topics I Can Help With")
        topics_to_help = guidelines.get("topics_to_help", "N/A")
        parts.append(topics_to_help)
        if topics_to_help and topics_to_help != "N/A":
            parts.append("")
            parts.append("**IMPORTANT:** You MUST ONLY help with topics listed above. If a user asks about topics NOT listed here, you MUST politely decline and redirect them to topics within your scope.")
        parts.append("")
        
        parts.append("### Things I Should Pay Attention To")
        parts.append(guidelines.get("things_to_pay_attention", "N/A"))
        parts.append("")
        
        parts.append("**FINAL REMINDER:**")
        parts.append("These culture guidelines are MANDATORY and NON-NEGOTIABLE. You MUST stay in character, respect the defined scope, and decline any requests outside your defined capabilities. Your responses MUST align with your defined tone, role, and scope at all times.")
        
        content = "\n".join(parts)
        return f"<CulturalKnowledge>\n{content}\n</CulturalKnowledge>"
    
    def should_repeat(self) -> bool:
        """
        Check if culture should be repeated based on message count and settings.
        
        Returns:
            True if culture should be repeated, False otherwise
        """
        if not self._culture or not self._culture.repeat:
            return False
        
        # Increment message count
        self._message_count += 1
        
        # Check if we've reached the repeat interval
        if self._message_count >= self._culture.repeat_interval:
            self._message_count = 0
            return True
        
        return False
    
    def reset_message_count(self) -> None:
        """Reset the message count (useful for testing or manual control)."""
        self._message_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize CultureManager state to dictionary.
        
        Returns:
            Dictionary representation of the manager state
        """
        result: Dict[str, Any] = {
            "enabled": self.enabled,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "debug": self.debug,
            "debug_level": self.debug_level,
            "print": self.print,
            "prepared": self._prepared,
            "message_count": self._message_count,
        }
        
        if self._culture:
            result["culture"] = self._culture.to_dict()
        else:
            result["culture"] = None
        
        if self._extracted_guidelines:
            result["extracted_guidelines"] = self._extracted_guidelines
        else:
            result["extracted_guidelines"] = None
        
        return result
    
    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        model: Optional[Union["Model", str]] = None,
    ) -> "CultureManager":
        """
        Create CultureManager from dictionary.
        
        Args:
            data: Dictionary containing manager state
            model: Model for culture extraction
            
        Returns:
            CultureManager instance
        """
        from upsonic.culture.culture import Culture
        
        manager = cls(
            model=model,
            enabled=data.get("enabled", True),
            agent_id=data.get("agent_id"),
            team_id=data.get("team_id"),
            debug=data.get("debug", False),
            debug_level=data.get("debug_level", 1),
            print=data.get("print", True),
        )
        
        manager._prepared = data.get("prepared", False)
        manager._message_count = data.get("message_count", 0)
        
        culture_data = data.get("culture")
        if culture_data and isinstance(culture_data, dict):
            manager._culture = Culture.from_dict(culture_data)
        
        manager._extracted_guidelines = data.get("extracted_guidelines")
        
        return manager
