"""
Profanity Detection Policies
Uses Detoxify library for comprehensive toxic content and profanity detection.
Supports multiple models: original, unbiased, multilingual, original-small, unbiased-small
"""

import asyncio
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    Detoxify = None


class ProfanityRule(RuleBase):
    """
    Rule to detect profanity and toxic content using Detoxify library.
    
    Supports multiple Detoxify models:
    - 'original': BERT-based model trained on Toxic Comment Classification Challenge
    - 'unbiased': RoBERTa-based model trained to minimize unintended bias
    - 'multilingual': XLM-RoBERTa-based model for multilingual detection
    - 'original-small': Lightweight Albert-based version of original
    - 'unbiased-small': Lightweight Albert-based version of unbiased
    """
    
    name = "Profanity Detection Rule"
    description = "Detects profanity, toxic content, and inappropriate language using Detoxify ML models"
    language = "en"
    
    def __init__(
        self, 
        options: Optional[Dict[str, Any]] = None,
        model_name: str = "unbiased",
        device: Optional[str] = None
    ):
        super().__init__(options)
        
        if not DETOXIFY_AVAILABLE:
            raise ImportError(
                "detoxify is not installed. Install it with: "
                "uv sync --extra embeddings or pip install detoxify"
            )
        
        # Model configuration
        self.model_name = options.get("model_name", model_name) if options else model_name
        self.device = options.get("device", device) if options else device
        
        # Validate model name
        valid_models = ["original", "unbiased", "multilingual", "original-small", "unbiased-small"]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. "
                f"Must be one of: {', '.join(valid_models)}"
            )
        
        # Initialize Detoxify model (lazy loading)
        self._model: Optional[Detoxify] = None
        self._model_lock = asyncio.Lock()
    
    def _get_model(self) -> Detoxify:
        """Lazy load the Detoxify model"""
        if self._model is None:
            if self.device:
                self._model = Detoxify(self.model_name, device=self.device)
            else:
                self._model = Detoxify(self.model_name)
        return self._model
    
    def _get_toxicity_labels(self) -> List[str]:
        """Get the labels returned by the current model"""
        model = self._get_model()
        
        # Different models return different labels
        # Test with empty string to get label structure
        test_result = model.predict("")
        
        # Return all keys except any metadata keys
        labels = [key for key in test_result.keys() if not key.startswith("_")]
        return labels
    
    def _get_all_category_scores(self, results: Dict[str, float]) -> List[str]:
        """Get all category scores from Detoxify results in format 'category:score'"""
        scores = []
        for label, score in results.items():
            if not label.startswith("_"):  # Skip metadata keys
                scores.append(f"{label}:{score:.6f}")
        return scores
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for profanity and toxic content detection"""
        
        if not policy_input.input_texts:
            return RuleOutput(
                confidence=0.0,
                content_type="SAFE_CONTENT",
                details="No input text provided"
            )
        
        try:
            model = self._get_model()
            
            # Detoxify can handle both single strings and lists
            input_texts = policy_input.input_texts
            
            # Get predictions from Detoxify
            # For single string: returns dict[str, float]
            # For list: returns dict[str, List[float]] where each key is a label
            if len(input_texts) == 1:
                results = model.predict(input_texts[0])
            else:
                results = model.predict(input_texts)
            
            # Check if results are batch format (dict with lists as values)
            is_batch = (
                isinstance(results, dict) and 
                len(results) > 0 and 
                isinstance(next(iter(results.values())), list)
            )
            
            if is_batch:
                # Batch results: results is dict[str, List[float]]
                # Process each text's scores and collect all scores
                num_texts = len(next(iter(results.values())))
                all_scores = []
                
                for i in range(num_texts):
                    # Extract scores for this text
                    text_results = {label: scores[i] for label, scores in results.items()}
                    # Get all category scores for this text
                    text_scores = self._get_all_category_scores(text_results)
                    all_scores.extend(text_scores)
                
                # Return all scores - Action will filter and compare
                max_score = max(
                    [float(score.split(":")[1]) for score in all_scores] if all_scores else [0.0]
                )
                
                return RuleOutput(
                    confidence=max_score,  # Max score for reference, but Action will decide
                    content_type="PROFANITY_DETECTION_RESULT",  # Neutral type, Action will decide
                    details=f"Detected {len(all_scores)} category scores across {num_texts} texts",
                    triggered_keywords=all_scores
                )
            else:
                # Single prediction result: results is dict[str, float]
                # Return all raw category scores - Action will filter and compare
                all_scores = self._get_all_category_scores(results)
                
                max_score = max(
                    [float(score.split(":")[1]) for score in all_scores] if all_scores else [0.0]
                )
                
                return RuleOutput(
                    confidence=max_score,  # Max score for reference, but Action will decide
                    content_type="PROFANITY_DETECTION_RESULT",  # Neutral type, Action will decide
                    details=f"Detected {len(all_scores)} category scores",
                    triggered_keywords=all_scores
                )
                
        except Exception as e:
            # Fallback: return low confidence but don't block
            return RuleOutput(
                confidence=0.1,
                content_type="SAFE_CONTENT",
                details=f"Error during profanity detection: {str(e)}. Content allowed with low confidence.",
                triggered_keywords=[]
            )
    
    async def process_async(self, policy_input: PolicyInput) -> RuleOutput:
        """Async variant of process for non-blocking execution"""
        # Run the synchronous process in a thread to avoid blocking
        return await asyncio.to_thread(self.process, policy_input)


class ProfanityBlockAction(ActionBase):
    """Action to block profanity and toxic content with appropriate messaging"""
    
    name = "Profanity Block Action"
    description = "Blocks content containing profanity or toxic language with educational message"
    language = "en"
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
    
    def _parse_scores_and_get_max(self, rule_result: RuleOutput) -> tuple[float, List[str]]:
        """Parse scores from triggered_keywords, filter those above threshold, return max score and filtered categories"""
        if not rule_result.triggered_keywords:
            return 0.0, []
        
        filtered_scores = []
        for item in rule_result.triggered_keywords:
            if ":" in item:
                category, score_str = item.split(":", 1)
                try:
                    score = float(score_str)
                    if score >= self.min_confidence:
                        filtered_scores.append((category, score))
                except (ValueError, TypeError):
                    continue
        
        if not filtered_scores:
            return 0.0, []
        
        # Get max score
        max_score = max(score for _, score in filtered_scores)
        # Get all categories that match max score
        max_categories = [f"{cat}:{score:.3f}" for cat, score in filtered_scores if score == max_score]
        
        return max_score, max_categories
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for profanity/toxic content"""
        
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return self.allow_content()
        
        # Block with educational message
        block_message = (
            f"This content has been blocked as it contains profanity, toxic language, or inappropriate "
            f"content (max toxicity score: {max_score:.3f}) that violates our community guidelines. "
            f"Please ensure your communication is respectful and appropriate for all audiences. "
            f"We encourage constructive and civil discourse."
        )
        
        return self.raise_block_error(block_message)
    
    async def action_async(self, rule_result: RuleOutput) -> PolicyOutput:
        """Async variant of action"""
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return await self.allow_content_async()
        
        block_message = (
            f"This content has been blocked as it contains profanity, toxic language, or inappropriate "
            f"content (max toxicity score: {max_score:.3f}) that violates our community guidelines. "
            f"Please ensure your communication is respectful and appropriate for all audiences. "
            f"We encourage constructive and civil discourse."
        )
        
        return await self.raise_block_error_async(block_message)


class ProfanityBlockAction_LLM(ActionBase):
    """LLM-powered action to block profanity with contextual messaging"""
    
    name = "Profanity Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for profanity/toxic content"
    language = "en"
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
    
    def _parse_scores_and_get_max(self, rule_result: RuleOutput) -> tuple[float, List[str]]:
        """Parse scores from triggered_keywords, filter those above threshold, return max score and filtered categories"""
        if not rule_result.triggered_keywords:
            return 0.0, []
        
        filtered_scores = []
        for item in rule_result.triggered_keywords:
            if ":" in item:
                category, score_str = item.split(":", 1)
                try:
                    score = float(score_str)
                    if score >= self.min_confidence:
                        filtered_scores.append((category, score))
                except (ValueError, TypeError):
                    continue
        
        if not filtered_scores:
            return 0.0, []
        
        # Get max score
        max_score = max(score for _, score in filtered_scores)
        # Get all categories that match max score
        max_categories = [f"{cat}:{score:.3f}" for cat, score in filtered_scores if score == max_score]
        
        return max_score, max_categories
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for profanity/toxic content"""
        
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = (
            f"Content contains profanity or toxic language. "
            f"Max toxicity score: {max_score:.3f}. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return self.llm_raise_block_error(reason)
    
    async def action_async(self, rule_result: RuleOutput) -> PolicyOutput:
        """Async variant of LLM action"""
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return await self.allow_content_async()
        
        reason = (
            f"Content contains profanity or toxic language. "
            f"Max toxicity score: {max_score:.3f}. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return await self.llm_raise_block_error_async(reason)


class ProfanityRaiseExceptionAction(ActionBase):
    """Action to raise exception for profanity/toxic content"""
    
    name = "Profanity Raise Exception Action"
    description = "Raises DisallowedOperation exception for profanity/toxic content"
    language = "en"
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
    
    def _parse_scores_and_get_max(self, rule_result: RuleOutput) -> tuple[float, List[str]]:
        """Parse scores from triggered_keywords, filter those above threshold, return max score and filtered categories"""
        if not rule_result.triggered_keywords:
            return 0.0, []
        
        filtered_scores = []
        for item in rule_result.triggered_keywords:
            if ":" in item:
                category, score_str = item.split(":", 1)
                try:
                    score = float(score_str)
                    if score >= self.min_confidence:
                        filtered_scores.append((category, score))
                except (ValueError, TypeError):
                    continue
        
        if not filtered_scores:
            return 0.0, []
        
        # Get max score
        max_score = max(score for _, score in filtered_scores)
        # Get all categories that match max score
        max_categories = [f"{cat}:{score:.3f}" for cat, score in filtered_scores if score == max_score]
        
        return max_score, max_categories
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for profanity/toxic content"""
        
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return self.allow_content()
        
        exception_message = (
            f"DisallowedOperation: Content contains profanity or toxic language "
            f"(max toxicity score: {max_score:.3f}) that violates platform policies. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return self.raise_exception(exception_message)
    
    async def action_async(self, rule_result: RuleOutput) -> PolicyOutput:
        """Async variant of raise exception action"""
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return await self.allow_content_async()
        
        exception_message = (
            f"DisallowedOperation: Content contains profanity or toxic language "
            f"(max toxicity score: {max_score:.3f}) that violates platform policies. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return self.raise_exception(exception_message)


class ProfanityRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for profanity/toxic content"""
    
    name = "Profanity Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for profanity/toxic content"
    language = "en"
    
    def __init__(self, min_confidence: float = 0.5):
        super().__init__()
        self.min_confidence = min_confidence
    
    def _parse_scores_and_get_max(self, rule_result: RuleOutput) -> tuple[float, List[str]]:
        """Parse scores from triggered_keywords, filter those above threshold, return max score and filtered categories"""
        if not rule_result.triggered_keywords:
            return 0.0, []
        
        filtered_scores = []
        for item in rule_result.triggered_keywords:
            if ":" in item:
                category, score_str = item.split(":", 1)
                try:
                    score = float(score_str)
                    if score >= self.min_confidence:
                        filtered_scores.append((category, score))
                except (ValueError, TypeError):
                    continue
        
        if not filtered_scores:
            return 0.0, []
        
        # Get max score
        max_score = max(score for _, score in filtered_scores)
        # Get all categories that match max score
        max_categories = [f"{cat}:{score:.3f}" for cat, score in filtered_scores if score == max_score]
        
        return max_score, max_categories
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for profanity/toxic content"""
        
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return self.allow_content()
        
        reason = (
            f"Content contains profanity or toxic language. "
            f"Max toxicity score: {max_score:.3f}. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return self.llm_raise_exception(reason)
    
    async def action_async(self, rule_result: RuleOutput) -> PolicyOutput:
        """Async variant of LLM raise exception action"""
        # Parse scores and get max score above threshold
        max_score, filtered_categories = self._parse_scores_and_get_max(rule_result)
        
        if max_score < self.min_confidence:
            return await self.allow_content_async()
        
        reason = (
            f"Content contains profanity or toxic language. "
            f"Max toxicity score: {max_score:.3f}. "
            f"Categories above threshold: {', '.join(filtered_categories) if filtered_categories else 'none'}. "
            f"Details: {rule_result.details}"
        )
        return await self.llm_raise_exception_async(reason)


# Pre-built Policies
# Only create policies if detoxify is available

if DETOXIFY_AVAILABLE:
    # ============================================================================
    # BLOCK POLICIES - Standard Block Actions
    # ============================================================================

    ## Profanity Block Policy (Unbiased Model) - Default
    # Standard policy using unbiased model to minimize bias in detection
    ProfanityBlockPolicy = Policy(
        name="Profanity Block Policy",
        description="Blocks content containing profanity or toxic language using Detoxify unbiased model",
        rule=ProfanityRule(model_name="unbiased"),
        action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Original Model)
    # Policy using original BERT-based model
    ProfanityBlockPolicy_Original = Policy(
    name="Profanity Block Policy (Original)",
    description="Blocks content containing profanity or toxic language using Detoxify original model",
    rule=ProfanityRule(model_name="original"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Multilingual Model)
    # Policy using multilingual model for multi-language support
    ProfanityBlockPolicy_Multilingual = Policy(
    name="Profanity Block Policy (Multilingual)",
    description="Blocks profanity/toxic content using Detoxify multilingual model (supports 7 languages)",
    rule=ProfanityRule(model_name="multilingual"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Original-Small Model)
    # Policy using lightweight Albert-based original model
    ProfanityBlockPolicy_OriginalSmall = Policy(
    name="Profanity Block Policy (Original-Small)",
    description="Blocks profanity/toxic content using Detoxify original-small lightweight model",
    rule=ProfanityRule(model_name="original-small"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Unbiased-Small Model)
    # Policy using lightweight Albert-based unbiased model
    ProfanityBlockPolicy_UnbiasedSmall = Policy(
    name="Profanity Block Policy (Unbiased-Small)",
    description="Blocks profanity/toxic content using Detoxify unbiased-small lightweight model",
    rule=ProfanityRule(model_name="unbiased-small"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    # ============================================================================
    # BLOCK POLICIES - LLM Block Actions
    # ============================================================================

    ## Profanity Block Policy with LLM (Unbiased Model)
    # Enhanced policy using LLM for better context understanding and blocking
    ProfanityBlockPolicy_LLM = Policy(
    name="Profanity Block Policy LLM",
    description="Uses LLM to generate contextual block messages for profanity/toxic content (unbiased model)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Original Model)
    ProfanityBlockPolicy_LLM_Original = Policy(
    name="Profanity Block Policy LLM (Original)",
    description="Uses LLM to generate contextual block messages using Detoxify original model",
    rule=ProfanityRule(model_name="original"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Multilingual Model)
    ProfanityBlockPolicy_LLM_Multilingual = Policy(
    name="Profanity Block Policy LLM (Multilingual)",
    description="Uses LLM to generate contextual block messages using Detoxify multilingual model",
    rule=ProfanityRule(model_name="multilingual"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Original-Small Model)
    ProfanityBlockPolicy_LLM_OriginalSmall = Policy(
    name="Profanity Block Policy LLM (Original-Small)",
    description="Uses LLM to generate contextual block messages using Detoxify original-small model",
    rule=ProfanityRule(model_name="original-small"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Unbiased-Small Model)
    ProfanityBlockPolicy_LLM_UnbiasedSmall = Policy(
    name="Profanity Block Policy LLM (Unbiased-Small)",
    description="Uses LLM to generate contextual block messages using Detoxify unbiased-small model",
    rule=ProfanityRule(model_name="unbiased-small"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    # ============================================================================
    # RAISE EXCEPTION POLICIES - Standard Exception Actions
    # ============================================================================

    ## Profanity Raise Exception Policy (Unbiased Model)
    # Policy that raises exceptions for profanity/toxic content
    ProfanityRaiseExceptionPolicy = Policy(
    name="Profanity Raise Exception Policy",
    description="Raises DisallowedOperation exception when profanity/toxic content is detected (unbiased model)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Original Model)
    ProfanityRaiseExceptionPolicy_Original = Policy(
    name="Profanity Raise Exception Policy (Original)",
    description="Raises DisallowedOperation exception using Detoxify original model",
    rule=ProfanityRule(model_name="original"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Multilingual Model)
    ProfanityRaiseExceptionPolicy_Multilingual = Policy(
    name="Profanity Raise Exception Policy (Multilingual)",
    description="Raises DisallowedOperation exception using Detoxify multilingual model",
    rule=ProfanityRule(model_name="multilingual"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Original-Small Model)
    ProfanityRaiseExceptionPolicy_OriginalSmall = Policy(
    name="Profanity Raise Exception Policy (Original-Small)",
    description="Raises DisallowedOperation exception using Detoxify original-small model",
    rule=ProfanityRule(model_name="original-small"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Unbiased-Small Model)
    ProfanityRaiseExceptionPolicy_UnbiasedSmall = Policy(
    name="Profanity Raise Exception Policy (Unbiased-Small)",
    description="Raises DisallowedOperation exception using Detoxify unbiased-small model",
    rule=ProfanityRule(model_name="unbiased-small"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    # ============================================================================
    # RAISE EXCEPTION POLICIES - LLM Exception Actions
    # ============================================================================

    ## Profanity Raise Exception Policy LLM (Unbiased Model)
    # Policy that uses LLM to generate contextual exception messages
    ProfanityRaiseExceptionPolicy_LLM = Policy(
    name="Profanity Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for profanity/toxic content (unbiased model)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Original Model)
    ProfanityRaiseExceptionPolicy_LLM_Original = Policy(
    name="Profanity Raise Exception Policy LLM (Original)",
    description="Raises DisallowedOperation exception with LLM-generated message using Detoxify original model",
    rule=ProfanityRule(model_name="original"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Multilingual Model)
    ProfanityRaiseExceptionPolicy_LLM_Multilingual = Policy(
    name="Profanity Raise Exception Policy LLM (Multilingual)",
    description="Raises DisallowedOperation exception with LLM-generated message using Detoxify multilingual model",
    rule=ProfanityRule(model_name="multilingual"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Original-Small Model)
    ProfanityRaiseExceptionPolicy_LLM_OriginalSmall = Policy(
    name="Profanity Raise Exception Policy LLM (Original-Small)",
    description="Raises DisallowedOperation exception with LLM-generated message using Detoxify original-small model",
    rule=ProfanityRule(model_name="original-small"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Unbiased-Small Model)
    ProfanityRaiseExceptionPolicy_LLM_UnbiasedSmall = Policy(
    name="Profanity Raise Exception Policy LLM (Unbiased-Small)",
    description="Raises DisallowedOperation exception with LLM-generated message using Detoxify unbiased-small model",
    rule=ProfanityRule(model_name="unbiased-small"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    # ============================================================================
    # THRESHOLD VARIATIONS - Common threshold values (0.3, 0.7)
    # ============================================================================

    ## Profanity Block Policy (Low Threshold - 0.3)
    # More sensitive detection with lower threshold
    ProfanityBlockPolicy_LowThreshold = Policy(
    name="Profanity Block Policy (Low Threshold)",
    description="Blocks profanity/toxic content with lower threshold (0.3) for more sensitive detection",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityBlockAction(min_confidence=0.3)
    )

    ## Profanity Block Policy (High Threshold - 0.7)
    # Less sensitive detection with higher threshold
    ProfanityBlockPolicy_HighThreshold = Policy(
    name="Profanity Block Policy (High Threshold)",
    description="Blocks profanity/toxic content with higher threshold (0.7) for less sensitive detection",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityBlockAction(min_confidence=0.7)
    )

    ## Profanity Block Policy LLM (Low Threshold - 0.3)
    ProfanityBlockPolicy_LLM_LowThreshold = Policy(
    name="Profanity Block Policy LLM (Low Threshold)",
    description="Uses LLM to block profanity/toxic content with lower threshold (0.3)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityBlockAction_LLM(min_confidence=0.3)
    )

    ## Profanity Block Policy LLM (High Threshold - 0.7)
    ProfanityBlockPolicy_LLM_HighThreshold = Policy(
    name="Profanity Block Policy LLM (High Threshold)",
    description="Uses LLM to block profanity/toxic content with higher threshold (0.7)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityBlockAction_LLM(min_confidence=0.7)
    )

    ## Profanity Raise Exception Policy (Low Threshold - 0.3)
    ProfanityRaiseExceptionPolicy_LowThreshold = Policy(
    name="Profanity Raise Exception Policy (Low Threshold)",
    description="Raises exception for profanity/toxic content with lower threshold (0.3)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.3)
    )

    ## Profanity Raise Exception Policy (High Threshold - 0.7)
    ProfanityRaiseExceptionPolicy_HighThreshold = Policy(
    name="Profanity Raise Exception Policy (High Threshold)",
    description="Raises exception for profanity/toxic content with higher threshold (0.7)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.7)
    )

    ## Profanity Raise Exception Policy LLM (Low Threshold - 0.3)
    ProfanityRaiseExceptionPolicy_LLM_LowThreshold = Policy(
    name="Profanity Raise Exception Policy LLM (Low Threshold)",
    description="Raises LLM-generated exception for profanity/toxic content with lower threshold (0.3)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.3)
    )

    ## Profanity Raise Exception Policy LLM (High Threshold - 0.7)
    ProfanityRaiseExceptionPolicy_LLM_HighThreshold = Policy(
    name="Profanity Raise Exception Policy LLM (High Threshold)",
    description="Raises LLM-generated exception for profanity/toxic content with higher threshold (0.7)",
    rule=ProfanityRule(model_name="unbiased"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.7)
    )

    # ============================================================================
    # DEVICE-SPECIFIC POLICIES - CPU device specification
    # ============================================================================

    ## Profanity Block Policy (CPU Device)
    # Explicitly uses CPU device
    ProfanityBlockPolicy_CPU = Policy(
    name="Profanity Block Policy (CPU)",
    description="Blocks profanity/toxic content using CPU device explicitly",
    rule=ProfanityRule(model_name="unbiased", device="cpu"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (CPU Device)
    ProfanityBlockPolicy_LLM_CPU = Policy(
    name="Profanity Block Policy LLM (CPU)",
    description="Uses LLM to block profanity/toxic content using CPU device",
    rule=ProfanityRule(model_name="unbiased", device="cpu"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (CPU Device)
    ProfanityRaiseExceptionPolicy_CPU = Policy(
    name="Profanity Raise Exception Policy (CPU)",
    description="Raises exception for profanity/toxic content using CPU device",
    rule=ProfanityRule(model_name="unbiased", device="cpu"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (CPU Device)
    ProfanityRaiseExceptionPolicy_LLM_CPU = Policy(
    name="Profanity Raise Exception Policy LLM (CPU)",
    description="Raises LLM-generated exception for profanity/toxic content using CPU device",
    rule=ProfanityRule(model_name="unbiased", device="cpu"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    # ============================================================================
    # GPU-SPECIFIC POLICIES - CUDA device specification
    # ============================================================================

    ## Profanity Block Policy (GPU/CUDA Device)
    # Explicitly uses CUDA device for GPU acceleration
    ProfanityBlockPolicy_GPU = Policy(
    name="Profanity Block Policy (GPU)",
    description="Blocks profanity/toxic content using GPU/CUDA device for faster processing",
    rule=ProfanityRule(model_name="unbiased", device="cuda"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Original Model - GPU)
    ProfanityBlockPolicy_Original_GPU = Policy(
    name="Profanity Block Policy (Original - GPU)",
    description="Blocks profanity/toxic content using Detoxify original model on GPU",
    rule=ProfanityRule(model_name="original", device="cuda"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Multilingual Model - GPU)
    ProfanityBlockPolicy_Multilingual_GPU = Policy(
    name="Profanity Block Policy (Multilingual - GPU)",
    description="Blocks profanity/toxic content using Detoxify multilingual model on GPU",
    rule=ProfanityRule(model_name="multilingual", device="cuda"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Original-Small Model - GPU)
    ProfanityBlockPolicy_OriginalSmall_GPU = Policy(
    name="Profanity Block Policy (Original-Small - GPU)",
    description="Blocks profanity/toxic content using Detoxify original-small model on GPU",
    rule=ProfanityRule(model_name="original-small", device="cuda"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy (Unbiased-Small Model - GPU)
    ProfanityBlockPolicy_UnbiasedSmall_GPU = Policy(
    name="Profanity Block Policy (Unbiased-Small - GPU)",
    description="Blocks profanity/toxic content using Detoxify unbiased-small model on GPU",
    rule=ProfanityRule(model_name="unbiased-small", device="cuda"),
    action=ProfanityBlockAction(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (GPU Device)
    ProfanityBlockPolicy_LLM_GPU = Policy(
    name="Profanity Block Policy LLM (GPU)",
    description="Uses LLM to block profanity/toxic content using GPU/CUDA device",
    rule=ProfanityRule(model_name="unbiased", device="cuda"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Original Model - GPU)
    ProfanityBlockPolicy_LLM_Original_GPU = Policy(
    name="Profanity Block Policy LLM (Original - GPU)",
    description="Uses LLM to block profanity/toxic content using Detoxify original model on GPU",
    rule=ProfanityRule(model_name="original", device="cuda"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Multilingual Model - GPU)
    ProfanityBlockPolicy_LLM_Multilingual_GPU = Policy(
    name="Profanity Block Policy LLM (Multilingual - GPU)",
    description="Uses LLM to block profanity/toxic content using Detoxify multilingual model on GPU",
    rule=ProfanityRule(model_name="multilingual", device="cuda"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Original-Small Model - GPU)
    ProfanityBlockPolicy_LLM_OriginalSmall_GPU = Policy(
    name="Profanity Block Policy LLM (Original-Small - GPU)",
    description="Uses LLM to block profanity/toxic content using Detoxify original-small model on GPU",
    rule=ProfanityRule(model_name="original-small", device="cuda"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Block Policy LLM (Unbiased-Small Model - GPU)
    ProfanityBlockPolicy_LLM_UnbiasedSmall_GPU = Policy(
    name="Profanity Block Policy LLM (Unbiased-Small - GPU)",
    description="Uses LLM to block profanity/toxic content using Detoxify unbiased-small model on GPU",
    rule=ProfanityRule(model_name="unbiased-small", device="cuda"),
    action=ProfanityBlockAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (GPU Device)
    ProfanityRaiseExceptionPolicy_GPU = Policy(
    name="Profanity Raise Exception Policy (GPU)",
    description="Raises exception for profanity/toxic content using GPU/CUDA device",
    rule=ProfanityRule(model_name="unbiased", device="cuda"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Original Model - GPU)
    ProfanityRaiseExceptionPolicy_Original_GPU = Policy(
    name="Profanity Raise Exception Policy (Original - GPU)",
    description="Raises exception for profanity/toxic content using Detoxify original model on GPU",
    rule=ProfanityRule(model_name="original", device="cuda"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Multilingual Model - GPU)
    ProfanityRaiseExceptionPolicy_Multilingual_GPU = Policy(
    name="Profanity Raise Exception Policy (Multilingual - GPU)",
    description="Raises exception for profanity/toxic content using Detoxify multilingual model on GPU",
    rule=ProfanityRule(model_name="multilingual", device="cuda"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Original-Small Model - GPU)
    ProfanityRaiseExceptionPolicy_OriginalSmall_GPU = Policy(
    name="Profanity Raise Exception Policy (Original-Small - GPU)",
    description="Raises exception for profanity/toxic content using Detoxify original-small model on GPU",
    rule=ProfanityRule(model_name="original-small", device="cuda"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy (Unbiased-Small Model - GPU)
    ProfanityRaiseExceptionPolicy_UnbiasedSmall_GPU = Policy(
    name="Profanity Raise Exception Policy (Unbiased-Small - GPU)",
    description="Raises exception for profanity/toxic content using Detoxify unbiased-small model on GPU",
    rule=ProfanityRule(model_name="unbiased-small", device="cuda"),
    action=ProfanityRaiseExceptionAction(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (GPU Device)
    ProfanityRaiseExceptionPolicy_LLM_GPU = Policy(
    name="Profanity Raise Exception Policy LLM (GPU)",
    description="Raises LLM-generated exception for profanity/toxic content using GPU/CUDA device",
    rule=ProfanityRule(model_name="unbiased", device="cuda"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Original Model - GPU)
    ProfanityRaiseExceptionPolicy_LLM_Original_GPU = Policy(
    name="Profanity Raise Exception Policy LLM (Original - GPU)",
    description="Raises LLM-generated exception using Detoxify original model on GPU",
    rule=ProfanityRule(model_name="original", device="cuda"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Multilingual Model - GPU)
    ProfanityRaiseExceptionPolicy_LLM_Multilingual_GPU = Policy(
    name="Profanity Raise Exception Policy LLM (Multilingual - GPU)",
    description="Raises LLM-generated exception using Detoxify multilingual model on GPU",
    rule=ProfanityRule(model_name="multilingual", device="cuda"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Original-Small Model - GPU)
    ProfanityRaiseExceptionPolicy_LLM_OriginalSmall_GPU = Policy(
    name="Profanity Raise Exception Policy LLM (Original-Small - GPU)",
    description="Raises LLM-generated exception using Detoxify original-small model on GPU",
    rule=ProfanityRule(model_name="original-small", device="cuda"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )

    ## Profanity Raise Exception Policy LLM (Unbiased-Small Model - GPU)
    ProfanityRaiseExceptionPolicy_LLM_UnbiasedSmall_GPU = Policy(
    name="Profanity Raise Exception Policy LLM (Unbiased-Small - GPU)",
    description="Raises LLM-generated exception using Detoxify unbiased-small model on GPU",
    rule=ProfanityRule(model_name="unbiased-small", device="cuda"),
    action=ProfanityRaiseExceptionAction_LLM(min_confidence=0.5)
    )
else:
    # If detoxify is not available, set all policies to None
    # This allows the module to be imported without errors
    ProfanityBlockPolicy = None
    ProfanityBlockPolicy_Original = None
    ProfanityBlockPolicy_Multilingual = None
    ProfanityBlockPolicy_OriginalSmall = None
    ProfanityBlockPolicy_UnbiasedSmall = None
    ProfanityBlockPolicy_LLM = None
    ProfanityBlockPolicy_LLM_Original = None
    ProfanityBlockPolicy_LLM_Multilingual = None
    ProfanityBlockPolicy_LLM_OriginalSmall = None
    ProfanityBlockPolicy_LLM_UnbiasedSmall = None
    ProfanityBlockPolicy_LowThreshold = None
    ProfanityBlockPolicy_HighThreshold = None
    ProfanityBlockPolicy_LLM_LowThreshold = None
    ProfanityBlockPolicy_LLM_HighThreshold = None
    ProfanityBlockPolicy_CPU = None
    ProfanityBlockPolicy_LLM_CPU = None
    ProfanityBlockPolicy_GPU = None
    ProfanityBlockPolicy_Original_GPU = None
    ProfanityBlockPolicy_Multilingual_GPU = None
    ProfanityBlockPolicy_OriginalSmall_GPU = None
    ProfanityBlockPolicy_UnbiasedSmall_GPU = None
    ProfanityBlockPolicy_LLM_GPU = None
    ProfanityBlockPolicy_LLM_Original_GPU = None
    ProfanityBlockPolicy_LLM_Multilingual_GPU = None
    ProfanityBlockPolicy_LLM_OriginalSmall_GPU = None
    ProfanityBlockPolicy_LLM_UnbiasedSmall_GPU = None
    ProfanityRaiseExceptionPolicy = None
    ProfanityRaiseExceptionPolicy_Original = None
    ProfanityRaiseExceptionPolicy_Multilingual = None
    ProfanityRaiseExceptionPolicy_OriginalSmall = None
    ProfanityRaiseExceptionPolicy_UnbiasedSmall = None
    ProfanityRaiseExceptionPolicy_LLM = None
    ProfanityRaiseExceptionPolicy_LLM_Original = None
    ProfanityRaiseExceptionPolicy_LLM_Multilingual = None
    ProfanityRaiseExceptionPolicy_LLM_OriginalSmall = None
    ProfanityRaiseExceptionPolicy_LLM_UnbiasedSmall = None
    ProfanityRaiseExceptionPolicy_LowThreshold = None
    ProfanityRaiseExceptionPolicy_HighThreshold = None
    ProfanityRaiseExceptionPolicy_LLM_LowThreshold = None
    ProfanityRaiseExceptionPolicy_LLM_HighThreshold = None
    ProfanityRaiseExceptionPolicy_CPU = None
    ProfanityRaiseExceptionPolicy_LLM_CPU = None
    ProfanityRaiseExceptionPolicy_GPU = None
    ProfanityRaiseExceptionPolicy_Original_GPU = None
    ProfanityRaiseExceptionPolicy_Multilingual_GPU = None
    ProfanityRaiseExceptionPolicy_OriginalSmall_GPU = None
    ProfanityRaiseExceptionPolicy_UnbiasedSmall_GPU = None
    ProfanityRaiseExceptionPolicy_LLM_GPU = None
    ProfanityRaiseExceptionPolicy_LLM_Original_GPU = None
    ProfanityRaiseExceptionPolicy_LLM_Multilingual_GPU = None
    ProfanityRaiseExceptionPolicy_LLM_OriginalSmall_GPU = None
    ProfanityRaiseExceptionPolicy_LLM_UnbiasedSmall_GPU = None

