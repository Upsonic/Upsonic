import json
import re
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

from upsonic.models.model_registry import (
    MODEL_REGISTRY,
    ModelMetadata,
    ModelCapability,
    ModelTier,
    get_models_by_capability,
    get_models_by_tier,
    get_top_models,
)


class ModelRecommendation(BaseModel):
    """Model recommendation with reasoning and confidence score."""
    
    model_name: str = Field(..., description="The recommended model identifier")
    reason: str = Field(..., description="Explanation for why this model was selected")
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the recommendation (0.0 to 1.0)"
    )
    alternative_models: List[str] = Field(
        default_factory=list,
        description="Alternative models that could also work"
    )
    estimated_cost_tier: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Relative cost tier (1=cheapest, 10=most expensive)"
    )
    estimated_speed_tier: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Relative speed tier (1=slowest, 10=fastest)"
    )
    selection_method: Literal["llm", "rule_based"] = Field(
        ...,
        description="Method used for selection"
    )


class SelectionCriteria(BaseModel):
    """Criteria for model selection."""
    
    # Task type indicators
    requires_reasoning: Optional[bool] = None
    requires_code_generation: Optional[bool] = None
    requires_math: Optional[bool] = None
    requires_creative_writing: Optional[bool] = None
    requires_vision: Optional[bool] = None
    requires_audio: Optional[bool] = None
    requires_long_context: Optional[bool] = None
    
    # Performance requirements
    prioritize_speed: bool = False
    prioritize_cost: bool = False
    prioritize_quality: bool = False
    
    # Constraints
    max_cost_tier: Optional[int] = None  # 1-10
    min_context_window: Optional[int] = None
    
    # Preferences
    preferred_provider: Optional[str] = None
    require_open_source: bool = False
    require_production_ready: bool = False
    
    # Specific capabilities
    required_capabilities: List[ModelCapability] = Field(default_factory=list)


class RuleBasedSelector:
    """Rule-based model selector that doesn't require an LLM."""
    
    def __init__(self):
        self.capability_keywords = {
            ModelCapability.REASONING: [
                "reason", "think", "analyze", "complex", "difficult",
                "problem solving", "logic", "inference", "deduce"
            ],
            ModelCapability.CODE_GENERATION: [
                "code", "programming", "function", "algorithm", "debug",
                "python", "javascript", "java", "script", "implement"
            ],
            ModelCapability.MATHEMATICS: [
                "math", "calculate", "equation", "formula", "statistics",
                "algebra", "geometry", "calculus", "proof"
            ],
            ModelCapability.CREATIVE_WRITING: [
                "write", "story", "creative", "poem", "article", "blog",
                "content", "narrative", "essay"
            ],
            ModelCapability.VISION: [
                "image", "picture", "photo", "visual", "see", "look",
                "analyze image", "describe image"
            ],
            ModelCapability.AUDIO: [
                "audio", "sound", "music", "voice", "speech", "listen"
            ],
            ModelCapability.LONG_CONTEXT: [
                "long", "large document", "entire file", "full text",
                "many pages", "comprehensive"
            ],
        }
    
    def _analyze_task_description(self, task_description: str) -> Dict[ModelCapability, float]:
        """Analyze task description and assign scores to capabilities."""
        task_lower = task_description.lower()
        capability_scores: Dict[ModelCapability, float] = {}
        
        for capability, keywords in self.capability_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in task_lower:
                    score += 1.0
            
            # Normalize by number of keywords
            if score > 0:
                capability_scores[capability] = min(score / len(keywords) * 10, 1.0)
        
        return capability_scores
    
    def _score_model(
        self,
        model: ModelMetadata,
        capability_scores: Dict[ModelCapability, float],
        criteria: Optional[SelectionCriteria] = None
    ) -> float:
        """Score a model based on task requirements and criteria."""
        score = 0.0
        
        # Score based on capability match
        for capability, weight in capability_scores.items():
            if capability in model.capabilities:
                score += weight * 100  # Weight capabilities heavily
        
        # Apply criteria if provided
        if criteria:
            # Check required capabilities
            if criteria.required_capabilities:
                if not all(cap in model.capabilities for cap in criteria.required_capabilities):
                    return 0.0  # Hard requirement not met
            
            # Check specific capability requirements (boolean fields)
            if criteria.requires_reasoning:
                if ModelCapability.REASONING not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_code_generation:
                if ModelCapability.CODE_GENERATION not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_math:
                if ModelCapability.MATHEMATICS not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_creative_writing:
                if ModelCapability.CREATIVE_WRITING not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_vision:
                if ModelCapability.VISION not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_audio:
                if ModelCapability.AUDIO not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            if criteria.requires_long_context:
                if ModelCapability.LONG_CONTEXT not in model.capabilities:
                    return 0.0  # Hard requirement not met
                score += 50  # Bonus for having required capability
            
            # Cost considerations
            if criteria.prioritize_cost:
                # Lower cost = higher score
                score += (10 - model.cost_tier) * 10
            
            if criteria.max_cost_tier and model.cost_tier > criteria.max_cost_tier:
                return 0.0  # Exceeds budget
            
            # Speed considerations
            if criteria.prioritize_speed:
                score += model.speed_tier * 10
            
            # Quality considerations
            if criteria.prioritize_quality:
                if model.benchmarks:
                    score += model.benchmarks.overall_score() * 2
            
            # Context window requirement
            if criteria.min_context_window:
                if model.context_window < criteria.min_context_window:
                    return 0.0  # Insufficient context
                elif model.context_window >= criteria.min_context_window:
                    score += 20  # Bonus for meeting requirement
            
            # Provider preference
            if criteria.preferred_provider:
                if model.provider == criteria.preferred_provider:
                    score += 50
            
            # Open source requirement
            if criteria.require_open_source:
                open_source_providers = ["meta", "qwen", "mistral"]
                if model.provider not in open_source_providers:
                    return 0.0
            
            # Production readiness
            if criteria.require_production_ready:
                if ModelCapability.PRODUCTION not in model.capabilities:
                    score *= 0.5  # Penalize but don't exclude
        
        # Add baseline tier bonus
        tier_scores = {
            ModelTier.FLAGSHIP: 100,
            ModelTier.ADVANCED: 80,
            ModelTier.STANDARD: 60,
            ModelTier.FAST: 40,
            ModelTier.SPECIALIZED: 70,
        }
        score += tier_scores.get(model.tier, 50)
        
        # Add benchmark bonus
        if model.benchmarks:
            score += model.benchmarks.overall_score() * 0.5
        
        return score
    
    def select_model(
        self,
        task_description: str,
        criteria: Optional[SelectionCriteria] = None,
        default_model: str = "openai/gpt-4o"
    ) -> ModelRecommendation:
        """
        Select the best model using rule-based logic.
        
        Args:
            task_description: Description of the task
            criteria: Optional selection criteria
            default_model: Fallback model if no good match found
        
        Returns:
            ModelRecommendation with selected model and reasoning
        """
        # Analyze task
        capability_scores = self._analyze_task_description(task_description)
        
        # Score all models
        model_scores: List[tuple[str, ModelMetadata, float]] = []
        for model_name, model in MODEL_REGISTRY.items():
            # Skip duplicate entries
            if "/" not in model_name:
                continue
            
            score = self._score_model(model, capability_scores, criteria)
            if score > 0:
                model_scores.append((model_name, model, score))
        
        if not model_scores:
            # No models matched criteria, use default
            default_meta = MODEL_REGISTRY.get(default_model)
            return ModelRecommendation(
                model_name=default_model,
                reason="No models matched the criteria. Using default model.",
                confidence_score=0.5,
                alternative_models=[],
                estimated_cost_tier=default_meta.cost_tier if default_meta else 5,
                estimated_speed_tier=default_meta.speed_tier if default_meta else 5,
                selection_method="rule_based"
            )
        
        # Sort by score
        model_scores.sort(key=lambda x: x[2], reverse=True)
        
        best_model_name, best_model, best_score = model_scores[0]
        alternatives = [name for name, _, _ in model_scores[1:6]]  # Top 5 alternatives
        
        # Calculate confidence based on score gap
        if len(model_scores) > 1:
            second_score = model_scores[1][2]
            score_gap = best_score - second_score
            confidence = min(0.7 + (score_gap / best_score) * 0.3, 0.99)
        else:
            confidence = 0.85
        
        # Build reasoning
        reasons = []
        if capability_scores:
            top_capabilities = sorted(
                capability_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            cap_names = [cap.value for cap, _ in top_capabilities]
            reasons.append(f"Task requires: {', '.join(cap_names)}")
        
        if best_model.benchmarks:
            reasons.append(
                f"Strong performance (overall score: {best_model.benchmarks.overall_score():.1f})"
            )
        
        if criteria:
            if criteria.prioritize_cost:
                reasons.append(f"Cost-effective (tier {best_model.cost_tier}/10)")
            if criteria.prioritize_speed:
                reasons.append(f"Fast inference (tier {best_model.speed_tier}/10)")
        
        reasons.append(f"{best_model.tier.value.title()} tier model")
        
        return ModelRecommendation(
            model_name=best_model_name,
            reason="; ".join(reasons),
            confidence_score=confidence,
            alternative_models=alternatives,
            estimated_cost_tier=best_model.cost_tier,
            estimated_speed_tier=best_model.speed_tier,
            selection_method="rule_based"
        )


class LLMBasedSelector:
    """LLM-based model selector using GPT-4o for intelligent recommendations."""
    
    def __init__(self, agent: Optional[Any] = None):
        """
        Initialize LLM-based selector.
        
        Args:
            agent: Agent instance to use for LLM calls (will create one if not provided)
        """
        self.agent = agent
    
    async def select_model_async(
        self,
        task_description: str,
        criteria: Optional[SelectionCriteria] = None,
        default_model: str = "openai/gpt-4o"
    ) -> ModelRecommendation:
        """
        Select the best model using an LLM for analysis.
        
        Args:
            task_description: Description of the task
            criteria: Optional selection criteria
            default_model: Fallback model if needed
        
        Returns:
            ModelRecommendation with LLM-selected model and reasoning
        """
        # Prepare model information for LLM
        model_info = self._prepare_model_info_for_llm()
        
        # Build the prompt
        prompt = self._build_selection_prompt(
            task_description,
            model_info,
            criteria
        )
        
        # Get or create agent
        if self.agent is None:
            from upsonic.agent.agent import Agent
            from upsonic.tasks.tasks import Task
            
            selection_agent = Agent(
                model="openai/gpt-4o",
                name="Model Selector",
                system_prompt="You are an expert at selecting the most appropriate AI model for different tasks."
            )
        else:
            selection_agent = self.agent
        
        # Create task
        from upsonic.tasks.tasks import Task
        
        selection_task = Task(
            description=prompt,
            response_format=ModelRecommendation,
            not_main_task=True
        )
        
        try:
            # Execute selection
            result = await selection_agent.do_async(selection_task)
            
            # Ensure result is a ModelRecommendation
            if isinstance(result, ModelRecommendation):
                result.selection_method = "llm"
                return result
            elif isinstance(result, dict):
                recommendation = ModelRecommendation(**result)
                recommendation.selection_method = "llm"
                return recommendation
            else:
                # Fallback to default
                default_meta = MODEL_REGISTRY.get(default_model)
                return ModelRecommendation(
                    model_name=default_model,
                    reason="LLM selection failed. Using default model.",
                    confidence_score=0.5,
                    alternative_models=[],
                    estimated_cost_tier=default_meta.cost_tier if default_meta else 5,
                    estimated_speed_tier=default_meta.speed_tier if default_meta else 5,
                    selection_method="llm"
                )
        except Exception as e:
            # Fallback to rule-based on error
            rule_selector = RuleBasedSelector()
            recommendation = rule_selector.select_model(task_description, criteria, default_model)
            recommendation.reason = f"LLM selection failed ({str(e)}). Used rule-based fallback: {recommendation.reason}"
            return recommendation
    
    def _prepare_model_info_for_llm(self) -> str:
        """Prepare concise model information for the LLM prompt."""
        model_summaries = []
        
        # Group by provider
        providers: Dict[str, List[tuple[str, ModelMetadata]]] = {}
        for name, meta in MODEL_REGISTRY.items():
            if "/" not in name:  # Skip aliases without provider prefix
                continue
            if meta.provider not in providers:
                providers[meta.provider] = []
            providers[meta.provider].append((name, meta))
        
        # Create summary for each provider
        for provider, models in sorted(providers.items()):
            # Deduplicate models by taking unique ones
            seen_names = set()
            unique_models = []
            for name, meta in models:
                if meta.name not in seen_names:
                    seen_names.add(meta.name)
                    unique_models.append((name, meta))
            
            provider_summary = [f"\n**{provider.upper()} Models:**"]
            
            for name, meta in unique_models:
                caps = ", ".join([c.value.replace("_", " ") for c in meta.capabilities[:5]])
                bench_summary = ""
                if meta.benchmarks:
                    bench_summary = f" (MMLU: {meta.benchmarks.mmlu:.1f}, Code: {meta.benchmarks.humaneval:.1f})" if meta.benchmarks.mmlu and meta.benchmarks.humaneval else ""
                
                model_summary = (
                    f"- **{name}**: {meta.tier.value} tier | "
                    f"Capabilities: {caps} | "
                    f"Context: {meta.context_window//1000}K tokens | "
                    f"Cost: {meta.cost_tier}/10 | Speed: {meta.speed_tier}/10"
                    f"{bench_summary}"
                )
                provider_summary.append(model_summary)
            
            model_summaries.append("\n".join(provider_summary))
        
        return "\n".join(model_summaries)
    
    def _build_selection_prompt(
        self,
        task_description: str,
        model_info: str,
        criteria: Optional[SelectionCriteria]
    ) -> str:
        """Build the prompt for LLM-based model selection."""
        criteria_text = ""
        if criteria:
            criteria_parts = []
            if criteria.prioritize_cost:
                criteria_parts.append("- Prioritize cost-effectiveness")
            if criteria.prioritize_speed:
                criteria_parts.append("- Prioritize inference speed")
            if criteria.prioritize_quality:
                criteria_parts.append("- Prioritize output quality")
            if criteria.max_cost_tier:
                criteria_parts.append(f"- Maximum cost tier: {criteria.max_cost_tier}/10")
            if criteria.min_context_window:
                criteria_parts.append(f"- Minimum context window: {criteria.min_context_window} tokens")
            if criteria.preferred_provider:
                criteria_parts.append(f"- Preferred provider: {criteria.preferred_provider}")
            if criteria.require_open_source:
                criteria_parts.append("- Must be open-source")
            if criteria.required_capabilities:
                caps = ", ".join([c.value for c in criteria.required_capabilities])
                criteria_parts.append(f"- Required capabilities: {caps}")
            
            # Add boolean requirement fields
            if criteria.requires_reasoning:
                criteria_parts.append("- REQUIRED: Advanced reasoning capabilities")
            if criteria.requires_code_generation:
                criteria_parts.append("- REQUIRED: Code generation capabilities")
            if criteria.requires_math:
                criteria_parts.append("- REQUIRED: Mathematical problem solving")
            if criteria.requires_creative_writing:
                criteria_parts.append("- REQUIRED: Creative writing capabilities")
            if criteria.requires_vision:
                criteria_parts.append("- REQUIRED: Vision/image processing capabilities")
            if criteria.requires_audio:
                criteria_parts.append("- REQUIRED: Audio processing capabilities")
            if criteria.requires_long_context:
                criteria_parts.append("- REQUIRED: Large context window support")
            
            if criteria_parts:
                criteria_text = "\n\n**Selection Criteria:**\n" + "\n".join(criteria_parts)
        
        prompt = f"""You are an expert AI model selector. Your job is to recommend the best AI model for a given task.

**Task Description:**
{task_description}
{criteria_text}

**Available Models:**
{model_info}

**Instructions:**
1. Analyze the task requirements carefully
2. Consider what capabilities are needed (reasoning, coding, math, vision, etc.)
3. Evaluate which models best match those needs
4. Consider the selection criteria if provided
5. Recommend the most suitable model with:
   - model_name: The full model identifier (e.g., "openai/gpt-4o")
   - reason: A clear explanation of why this model is the best choice
   - confidence_score: Your confidence in this recommendation (0.0 to 1.0)
   - alternative_models: 2-3 alternative models that could also work (list of model names)
   - estimated_cost_tier: The cost tier of the recommended model (1-10)
   - estimated_speed_tier: The speed tier of the recommended model (1-10)

**Important Guidelines:**
- Match model capabilities to task requirements
- Consider the performance tiers (flagship > advanced > standard > fast)
- Balance cost, speed, and quality based on criteria
- For complex reasoning tasks, prefer flagship or specialized reasoning models
- For code generation, prioritize models with high HumanEval scores
- For math problems, prioritize models with high MATH/GSM8K scores
- For cost-sensitive tasks, recommend fast/standard tier models
- For long documents, ensure sufficient context window

Return your recommendation as a structured response matching the ModelRecommendation format."""
        
        return prompt


def select_model(
    task_description: str,
    criteria: Optional[SelectionCriteria] = None,
    use_llm: bool = False,
    agent: Optional[Any] = None,
    default_model: str = "openai/gpt-4o"
) -> ModelRecommendation:
    """
    Select the best model for a task (synchronous wrapper).
    
    Args:
        task_description: Description of the task
        criteria: Optional selection criteria
        use_llm: Whether to use LLM-based selection
        agent: Agent instance for LLM calls
        default_model: Fallback model
    
    Returns:
        ModelRecommendation
    """
    import asyncio
    
    if use_llm:
        selector = LLMBasedSelector(agent)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use asyncio.run in running loop, fallback to rule-based
                selector = RuleBasedSelector()
                return selector.select_model(task_description, criteria, default_model)
            else:
                return loop.run_until_complete(
                    selector.select_model_async(task_description, criteria, default_model)
                )
        except RuntimeError:
            return asyncio.run(
                selector.select_model_async(task_description, criteria, default_model)
            )
    else:
        selector = RuleBasedSelector()
        return selector.select_model(task_description, criteria, default_model)


async def select_model_async(
    task_description: str,
    criteria: Optional[SelectionCriteria] = None,
    use_llm: bool = False,
    agent: Optional[Any] = None,
    default_model: str = "openai/gpt-4o"
) -> ModelRecommendation:
    """
    Select the best model for a task (asynchronous).
    
    Args:
        task_description: Description of the task
        criteria: Optional selection criteria
        use_llm: Whether to use LLM-based selection
        agent: Agent instance for LLM calls
        default_model: Fallback model
    
    Returns:
        ModelRecommendation
    """
    if use_llm:
        selector = LLMBasedSelector(agent)
        return await selector.select_model_async(task_description, criteria, default_model)
    else:
        selector = RuleBasedSelector()
        return selector.select_model(task_description, criteria, default_model)

