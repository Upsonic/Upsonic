"""
Unit tests for Profanity Detection Policies using Detoxify.

Unit tests focus on testing components directly without Agent integration:
- Rule detection with different models
- Threshold behavior
- Batch processing
- Action behaviors (block, exception)
- Policy execution
- Edge cases and error handling
"""

import pytest
from upsonic.safety_engine.models import PolicyInput, RuleOutput, PolicyOutput
from upsonic.safety_engine.policies.profanity_policies import (
    ProfanityRule,
    ProfanityBlockAction,
    ProfanityBlockAction_LLM,
    ProfanityRaiseExceptionAction,
    ProfanityRaiseExceptionAction_LLM,
    ProfanityBlockPolicy,
    ProfanityBlockPolicy_Original,
    ProfanityBlockPolicy_Multilingual,
    ProfanityBlockPolicy_LLM,
    ProfanityRaiseExceptionPolicy,
    ProfanityRaiseExceptionPolicy_LLM,
    DETOXIFY_AVAILABLE
)

# Skip all tests if detoxify is not available
pytestmark = pytest.mark.skipif(
    not DETOXIFY_AVAILABLE,
    reason="detoxify is not installed. Install with: uv sync --extra policy"
)


class TestProfanityRule:
    """Test the ProfanityRule class directly with real Detoxify models"""
    
    def test_rule_initialization_default(self):
        """Test rule initialization with default parameters"""
        rule = ProfanityRule()
        assert rule.model_name == "unbiased"
        assert rule.device is None
    
    def test_rule_initialization_custom(self):
        """Test rule initialization with custom parameters"""
        rule = ProfanityRule(
            model_name="original",
            device="cpu"
        )
        assert rule.model_name == "original"
        assert rule.device == "cpu"
    
    def test_rule_initialization_with_options(self):
        """Test rule initialization with options dict"""
        options = {
            "model_name": "multilingual",
            "device": "cpu"
        }
        rule = ProfanityRule(options=options)
        assert rule.model_name == "multilingual"
    
    def test_rule_invalid_model_name(self):
        """Test rule raises error for invalid model name"""
        with pytest.raises(ValueError, match="Invalid model_name"):
            ProfanityRule(model_name="invalid_model")
    
    def test_rule_detects_toxic_content(self):
        """Test rule detects toxic content using real Detoxify model"""
        rule = ProfanityRule(model_name="unbiased")
        
        # Use known toxic phrases that Detoxify will detect
        toxic_phrases = [
            "You are an idiot and I hate you!",
            "This is stupid and you're a moron!",
            "You're a worthless piece of trash!",
            "I hate you so much!",
        ]
        
        detected_toxic = False
        for toxic_phrase in toxic_phrases:
            policy_input = PolicyInput(input_texts=[toxic_phrase])
            result = rule.process(policy_input)
            
            assert isinstance(result, RuleOutput)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            assert result.content_type == "PROFANITY_DETECTION_RESULT"  # Rule returns neutral type
            assert result.triggered_keywords is not None
            
            # Check if any score is high (above 0.5 as typical threshold)
            for item in result.triggered_keywords:
                if ":" in item:
                    try:
                        score = float(item.split(":")[1])
                        if score >= 0.5:  # Check against typical action threshold
                            detected_toxic = True
                            break
                    except (ValueError, TypeError):
                        continue
            if detected_toxic:
                break
        
        # At least one toxic phrase should be detected
        assert detected_toxic, "At least one toxic phrase should be detected with high scores"
    
    def test_rule_allows_safe_content(self):
        """Test rule allows safe content using real Detoxify model"""
        rule = ProfanityRule(model_name="unbiased")
        
        safe_phrases = [
            "Hello, how are you today? I hope you're having a wonderful day.",
            "This is a friendly conversation about the weather.",
            "I would like to learn more about machine learning.",
            "Thank you for your help with this project.",
        ]
        
        for safe_phrase in safe_phrases:
            policy_input = PolicyInput(input_texts=[safe_phrase])
            result = rule.process(policy_input)
            
            assert isinstance(result, RuleOutput)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0
            
            # Rule returns all scores, content_type is neutral - Action will decide
            # Safe content should have low scores in triggered_keywords
            assert result.content_type == "PROFANITY_DETECTION_RESULT"
            assert result.triggered_keywords is not None
            # Check that all scores are low (below typical action threshold of 0.5)
            all_scores_low = True
            for item in result.triggered_keywords:
                if ":" in item:
                    try:
                        score = float(item.split(":")[1])
                        if score >= 0.5:  # Check against typical action threshold
                            all_scores_low = False
                            break
                    except (ValueError, TypeError):
                        continue
            # For safe content, all scores should be below typical threshold
            assert all_scores_low or result.confidence < 0.5
    
    def test_rule_batch_processing(self):
        """Test rule processes multiple texts in batch using real Detoxify"""
        rule = ProfanityRule(model_name="unbiased")
        
        policy_input = PolicyInput(
            input_texts=[
                "Hello, this is a friendly message.",
                "This is a toxic and hateful comment that should be detected!",
                "Another safe message here.",
                "You're an idiot!",
            ]
        )
        
        result = rule.process(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
        
        # Rule returns all scores, check if any are high
        assert result.content_type == "PROFANITY_DETECTION_RESULT"
        assert result.triggered_keywords is not None
        # Check if any score in batch is above typical threshold
        has_toxic = False
        for item in result.triggered_keywords:
            if ":" in item:
                try:
                    score = float(item.split(":")[1])
                    if score >= 0.5:  # Check against typical action threshold
                        has_toxic = True
                        break
                except (ValueError, TypeError):
                    continue
        # With toxic content in batch, should detect it
        assert has_toxic or result.confidence >= 0.5
    
    def test_rule_empty_input(self):
        """Test rule handles empty input gracefully"""
        rule = ProfanityRule()
        
        policy_input = PolicyInput(input_texts=[])
        
        result = rule.process(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence == 0.0
        assert result.content_type == "SAFE_CONTENT"
        assert "No input text" in result.details
    
    def test_rule_none_input(self):
        """Test rule handles None input gracefully"""
        rule = ProfanityRule()
        
        policy_input = PolicyInput(input_texts=None)
        
        result = rule.process(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence == 0.0
        assert result.content_type == "SAFE_CONTENT"
    
    @pytest.mark.asyncio
    async def test_rule_async_processing(self):
        """Test async processing of rule"""
        rule = ProfanityRule(model_name="unbiased")
        
        policy_input = PolicyInput(
            input_texts=["This is a test message for async processing."]
        )
        
        result = await rule.process_async(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
    
    def test_rule_different_models(self):
        """Test rule works with different model types using real models"""
        models = ["original", "unbiased"]
        test_text = "This is a test message for model validation."
        
        for model_name in models:
            rule = ProfanityRule(model_name=model_name)
            policy_input = PolicyInput(input_texts=[test_text])
            
            result = rule.process(policy_input)
            assert isinstance(result, RuleOutput)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0


class TestProfanityActions:
    """Test the Action classes with real rule results"""
    
    def test_block_action_allows_low_confidence(self):
        """Test block action allows content below threshold"""
        action = ProfanityBlockAction(min_confidence=0.5)
        
        rule_result = RuleOutput(
            confidence=0.3,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="Low confidence detection"
        )
        
        # Set original content
        action.original_content = ["Test content"]
        action.detected_language = "en"
        
        result = action.action(rule_result)
        
        assert isinstance(result, PolicyOutput)
        assert result.action_output["action_taken"] == "ALLOW"
    
    def test_block_action_blocks_high_confidence(self):
        """Test block action blocks content above threshold"""
        action = ProfanityBlockAction(min_confidence=0.5)
        
        rule_result = RuleOutput(
            confidence=0.8,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="High confidence toxic content",
            triggered_keywords=["toxicity:0.850", "insult:0.750", "threat:0.600"]
        )
        
        action.original_content = ["Toxic content"]
        action.detected_language = "en"
        
        result = action.action(rule_result)
        
        assert isinstance(result, PolicyOutput)
        assert result.action_output["action_taken"] == "BLOCK"
        assert "blocked" in result.action_output["message"].lower()
    
    @pytest.mark.asyncio
    async def test_block_action_async(self):
        """Test async block action"""
        action = ProfanityBlockAction(min_confidence=0.5)
        
        rule_result = RuleOutput(
            confidence=0.8,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="High confidence toxic content",
            triggered_keywords=["toxicity:0.850", "insult:0.750", "threat:0.600"]
        )
        
        action.original_content = ["Toxic content"]
        action.detected_language = "en"
        
        result = await action.action_async(rule_result)
        
        assert isinstance(result, PolicyOutput)
        assert result.action_output["action_taken"] == "BLOCK"
    
    def test_raise_exception_action_allows_low_confidence(self):
        """Test raise exception action allows content below threshold"""
        action = ProfanityRaiseExceptionAction(min_confidence=0.5)
        
        rule_result = RuleOutput(
            confidence=0.3,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="Low confidence detection"
        )
        
        action.original_content = ["Test content"]
        action.detected_language = "en"
        
        result = action.action(rule_result)
        
        assert isinstance(result, PolicyOutput)
        assert result.action_output["action_taken"] == "ALLOW"
    
    def test_raise_exception_action_raises_exception(self):
        """Test raise exception action raises exception for high confidence"""
        action = ProfanityRaiseExceptionAction(min_confidence=0.5)
        
        rule_result = RuleOutput(
            confidence=0.8,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="High confidence toxic content",
            triggered_keywords=["toxicity:0.850", "insult:0.750", "threat:0.600"]
        )
        
        action.original_content = ["Toxic content"]
        action.detected_language = "en"
        
        from upsonic.safety_engine.exceptions import DisallowedOperation
        
        with pytest.raises(DisallowedOperation):
            action.action(rule_result)
    
    def test_raise_exception_action_with_high_confidence(self):
        """Test ProfanityRaiseExceptionAction raises exception with high confidence rule result"""
        from upsonic.safety_engine.exceptions import DisallowedOperation
        from upsonic.safety_engine.policies.profanity_policies import ProfanityRaiseExceptionAction
        
        # Create action with lower threshold to ensure exception is raised
        action = ProfanityRaiseExceptionAction(min_confidence=0.3)
        
        # Create high confidence rule result with scores in triggered_keywords
        rule_result = RuleOutput(
            confidence=0.998,
            content_type="PROFANITY_TOXIC_CONTENT",
            details="High confidence toxic content detected",
            triggered_keywords=["toxicity:0.998", "insult:0.991", "threat:0.200"]
        )
        
        # Set original content
        action.original_content = ["You are an idiot and I hate you!"]
        action.detected_language = "en"
        
        # This should raise DisallowedOperation
        with pytest.raises(DisallowedOperation) as exc_info:
            action.action(rule_result)
        
        # Verify exception message
        exception_message = str(exc_info.value)
        assert "profanity" in exception_message.lower() or "toxic" in exception_message.lower()
        assert "0.998" in exception_message or "0.99" in exception_message


class TestProfanityPolicies:
    """Test the complete Policy instances with real Detoxify"""
    
    def test_profanity_block_policy_execution(self):
        """Test ProfanityBlockPolicy execution"""
        policy_input = PolicyInput(
            input_texts=["This is a test message."]
        )
        
        rule_result, action_result, policy_result = ProfanityBlockPolicy.execute(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
        assert isinstance(policy_result, PolicyOutput)
    
    def test_profanity_block_policy_original(self):
        """Test ProfanityBlockPolicy_Original execution"""
        policy_input = PolicyInput(
            input_texts=["Test message for original model."]
        )
        
        rule_result, action_result, policy_result = ProfanityBlockPolicy_Original.execute(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
    
    def test_profanity_block_policy_multilingual(self):
        """Test ProfanityBlockPolicy_Multilingual execution"""
        policy_input = PolicyInput(
            input_texts=["Test message for multilingual model."]
        )
        
        rule_result, action_result, policy_result = ProfanityBlockPolicy_Multilingual.execute(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
    
    @pytest.mark.asyncio
    async def test_profanity_block_policy_async(self):
        """Test async policy execution"""
        policy_input = PolicyInput(
            input_texts=["Test message for async execution."]
        )
        
        rule_result, action_result, policy_result = await ProfanityBlockPolicy.execute_async(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
        assert isinstance(policy_result, PolicyOutput)
    
    def test_profanity_raise_exception_policy_safe_content(self):
        """Test ProfanityRaiseExceptionPolicy allows safe content"""
        policy_input = PolicyInput(
            input_texts=["Safe test message."]
        )
        
        # Should not raise exception for safe content
        rule_result, action_result, policy_result = ProfanityRaiseExceptionPolicy.execute(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
    
    def test_profanity_raise_exception_policy_raises_exception(self):
        """Test ProfanityRaiseExceptionPolicy raises DisallowedOperation for toxic content"""
        from upsonic.safety_engine.exceptions import DisallowedOperation
        
        # Use known toxic phrases that Detoxify will detect
        toxic_phrases = [
            "You are an idiot and I hate you!",
            "This is stupid and you're a moron!",
            "You're a worthless piece of trash!",
        ]
        
        exception_raised = False
        for toxic_phrase in toxic_phrases:
            policy_input = PolicyInput(input_texts=[toxic_phrase])
            
            try:
                rule_result, action_result, policy_result = ProfanityRaiseExceptionPolicy.execute(policy_input)
                
                # Check if confidence is high enough to trigger exception
                if rule_result.confidence >= 0.5:
                    # If confidence is high, exception should have been raised
                    # But if it wasn't, the action might have allowed it
                    pass
            except DisallowedOperation as e:
                exception_raised = True
                assert "profanity" in str(e).lower() or "toxic" in str(e).lower()
                break
        
        # At least one should raise DisallowedOperation
        assert exception_raised, "DisallowedOperation should be raised for toxic content"


class TestProfanityPolicyEdgeCases:
    """Test edge cases and error handling"""
    
    def test_policy_with_custom_threshold(self):
        """Test policy with custom threshold"""
        from upsonic.safety_engine.policies.profanity_policies import ProfanityRule, ProfanityBlockAction
        from upsonic.safety_engine.base.policy import Policy
        
        custom_rule = ProfanityRule(model_name="unbiased")
        custom_action = ProfanityBlockAction(min_confidence=0.8)
        
        custom_policy = Policy(
            name="Custom Profanity Policy",
            description="Custom threshold profanity policy",
            rule=custom_rule,
            action=custom_action
        )
        
        policy_input = PolicyInput(
            input_texts=["Test message with custom threshold."]
        )
        
        rule_result, action_result, policy_result = custom_policy.execute(policy_input)
        
        assert isinstance(rule_result, RuleOutput)
        assert isinstance(action_result, PolicyOutput)
    
    def test_policy_with_different_models(self):
        """Test policies with different model types"""
        models = ["original", "unbiased"]
        
        for model_name in models:
            rule = ProfanityRule(model_name=model_name)
            action = ProfanityBlockAction(min_confidence=0.5)
            
            from upsonic.safety_engine.base.policy import Policy
            
            policy = Policy(
                name=f"Test Policy {model_name}",
                description=f"Test policy with {model_name} model",
                rule=rule,
                action=action
            )
            
            policy_input = PolicyInput(
                input_texts=[f"Test message for {model_name} model."]
            )
            
            rule_result, action_result, policy_result = policy.execute(policy_input)
            
            assert isinstance(rule_result, RuleOutput)
            assert isinstance(action_result, PolicyOutput)
    
    def test_policy_batch_processing_large(self):
        """Test policy handles large batch of texts"""
        rule = ProfanityRule(model_name="unbiased")
        
        # Create a batch of texts
        texts = [f"Test message number {i}." for i in range(10)]
        policy_input = PolicyInput(input_texts=texts)
        
        result = rule.process(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
    
    def test_policy_very_long_text(self):
        """Test policy handles very long text"""
        rule = ProfanityRule(model_name="unbiased")
        
        # Create a very long text
        long_text = "This is a test. " * 1000
        policy_input = PolicyInput(input_texts=[long_text])
        
        result = rule.process(policy_input)
        
        assert isinstance(result, RuleOutput)
        assert result.confidence >= 0.0
        assert result.confidence <= 1.0
