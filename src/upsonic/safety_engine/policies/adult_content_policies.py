"""
Adult Content Policies
Handles explicit material and adult themes.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class AdultContentRule(RuleBase):
    """Rule to detect adult content including explicit material and adult themes"""
    
    name = "Adult Content Rule"
    description = "Detects explicit sexual content, adult themes, and inappropriate material"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Explicit sexual keywords
        self.explicit_keywords = [
            # Direct sexual terms (using asterisks to avoid false positives in logs)
            "sex", "sexual", "porn", "pornography", "xxx", "adult video", "adult film",
            "explicit", "nude", "naked", "strip", "erotic", "orgasm", "climax",
            "masturbation", "masturbate", "intercourse", "penetration", "oral sex",
            "anal sex", "vaginal", "penis", "vagina", "genitals", "private parts",
            
            # Adult industry terms
            "adult entertainment", "adult content", "adult website", "cam girl",
            "cam boy", "webcam show", "live sex", "phone sex", "sexting",
            "adult chat", "hookup", "one night stand", "booty call", "fwb",
            
            # Suggestive terms
            "horny", "aroused", "turned on", "hot and bothered", "steamy",
            "seductive", "sensual", "provocative", "suggestive", "intimate",
            "passionate", "lustful", "desire", "temptation", "fantasy",
            
            # Adult content platforms/sites
            "pornhub", "xvideos", "onlyfans", "chaturbate", "cam4", "livejasmin",
            "adult friend finder", "ashley madison", "seeking arrangement",
            
            # Fetish and kink terms
            "fetish", "kink", "bdsm", "bondage", "dominance", "submission",
            "roleplay", "costume play", "lingerie", "underwear show",
            
            # Adult services
            "escort", "prostitution", "call girl", "massage parlor", "strip club",
            "gentleman's club", "adult services", "companionship services"
        ]
        
        # Adult content patterns
        self.adult_patterns = [
            r'\b(?:want to|wanna|let\'s)\s+(?:have sex|make love|get intimate|hook up|sleep together)',
            r'\b(?:send|show)\s+(?:me\s+)?(?:nudes|naked pics|explicit photos|dick pics)',
            r'\b(?:looking for|seeking)\s+(?:sex|hookup|casual encounter|adult fun)',
            r'\b(?:18\+|adults? only|mature content|explicit content|nsfw)',
            r'\b(?:click here for|visit)\s+(?:adult|porn|xxx|explicit)\s+(?:content|website|videos)',
            r'\b(?:hot|sexy|horny)\s+(?:singles?|girls?|boys?|women|men)\s+(?:in your area|near you|waiting)',
            r'\b(?:free|watch)\s+(?:porn|adult videos|xxx|explicit content)',
            r'\b(?:join|subscribe)\s+(?:adult site|porn site|xxx site|explicit content)',
            r'\b(?:rate|review)\s+(?:my|our)\s+(?:nudes|explicit photos|adult content)',
            r'\b(?:adult|mature|explicit)\s+(?:dating|chat|website|content|material)'
        ]
        
        # Suggestive but less explicit terms
        self.suggestive_keywords = [
            "flirt", "flirting", "tease", "teasing", "seduce", "seduction",
            "attraction", "chemistry", "spark", "tension", "romance",
            "dating", "relationship", "boyfriend", "girlfriend", "lover",
            "kiss", "kissing", "hug", "hugging", "cuddle", "cuddling",
            "touch", "touching", "caress", "massage", "embrace"
        ]
        
        # Age verification terms
        self.age_verification_keywords = [
            "18+", "21+", "adults only", "mature audiences", "age verification",
            "must be 18", "over 18", "adult age", "legal age", "age of consent"
        ]
        
        # Allow custom keywords from options
        if options:
            if "explicit_keywords" in options:
                self.explicit_keywords.extend(options["explicit_keywords"])
            if "adult_patterns" in options:
                self.adult_patterns.extend(options["adult_patterns"])
            if "suggestive_keywords" in options:
                self.suggestive_keywords.extend(options["suggestive_keywords"])
            if "age_verification_keywords" in options:
                self.age_verification_keywords.extend(options["age_verification_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for adult content detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or []).lower()
        
        # Find explicit content
        triggered_explicit = []
        for keyword in self.explicit_keywords:
            # Use word boundary pattern that handles plurals and variations
            pattern = r'\b' + re.escape(keyword.lower()) + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text):
                triggered_explicit.append(keyword)
        
        # Find adult content patterns
        triggered_patterns = []
        for pattern in self.adult_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                triggered_patterns.append(pattern)
        
        # Find suggestive content
        triggered_suggestive = []
        for keyword in self.suggestive_keywords:
            # Use word boundary pattern that handles plurals and variations
            pattern = r'\b' + re.escape(keyword.lower()) + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text):
                triggered_suggestive.append(keyword)
        
        # Find age verification terms
        triggered_age_verification = []
        for keyword in self.age_verification_keywords:
            # Use word boundary pattern that handles plurals and variations
            pattern = r'\b' + re.escape(keyword.lower()) + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text):
                triggered_age_verification.append(keyword)
        
        # Combine all triggered items
        all_triggered = triggered_explicit + triggered_patterns + triggered_suggestive + triggered_age_verification
        
        # Calculate confidence based on content severity
        if not all_triggered:
            return RuleOutput(
                confidence=0.0,
                content_type="SAFE_CONTENT",
                details="No adult content detected"
            )
        
        # Weight different types of content
        explicit_weight = len(triggered_explicit) * 0.9  # High weight for explicit content
        pattern_weight = len(triggered_patterns) * 0.85  # High weight for adult patterns
        suggestive_weight = len(triggered_suggestive) * 0.4  # Moderate weight for suggestive content
        age_verification_weight = len(triggered_age_verification) * 0.7  # Moderate-high weight
        
        total_weight = explicit_weight + pattern_weight + suggestive_weight + age_verification_weight
        confidence = min(1.0, total_weight)
        
        # Determine content type based on severity
        if triggered_explicit or triggered_patterns:
            content_type = "EXPLICIT_ADULT_CONTENT"
        elif triggered_age_verification:
            content_type = "AGE_RESTRICTED_CONTENT"
        else:
            content_type = "SUGGESTIVE_CONTENT"
        
        return RuleOutput(
            confidence=confidence,
            content_type=content_type,
            details=f"Detected {len(all_triggered)} adult content indicators: {len(triggered_explicit)} explicit terms, {len(triggered_patterns)} adult patterns, {len(triggered_suggestive)} suggestive terms, {len(triggered_age_verification)} age verification terms",
            triggered_keywords=all_triggered
        )


class AdultContentRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect adult content with better context understanding"""
    
    name = "Adult Content LLM Rule"
    description = "Uses LLM to detect adult content and explicit material with context awareness"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for adult content detection"""
        
        if not self.text_finder_llm:
            # Fallback to keyword-based detection
            fallback_rule = AdultContentRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find adult content
            triggered_keywords = self._llm_find_keywords_with_input("ADULT_CONTENT", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="SAFE_CONTENT",
                    details="No adult content detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="EXPLICIT_ADULT_CONTENT",
                details=f"LLM detected {len(triggered_keywords)} adult content indicators",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to keyword-based detection on error
            fallback_rule = AdultContentRule()
            return fallback_rule.process(policy_input)


class AdultContentBlockAction(ActionBase):
    """Action to block adult content with appropriate messaging"""
    
    name = "Adult Content Block Action"
    description = "Blocks content containing explicit material or adult themes"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for adult content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Different messages based on content type
        if rule_result.content_type == "EXPLICIT_ADULT_CONTENT":
            block_message = (
                "This content has been blocked as it contains explicit adult material "
                "that is not appropriate for this platform. Please ensure your content "
                "is suitable for all audiences."
            )
        elif rule_result.content_type == "AGE_RESTRICTED_CONTENT":
            block_message = (
                "This content has been blocked as it appears to be age-restricted material. "
                "Please verify that your content is appropriate for all users."
            )
        else:  # SUGGESTIVE_CONTENT
            block_message = (
                "This content has been blocked as it contains suggestive themes that "
                "may not be appropriate. Please review and modify your content."
            )
        
        return self.raise_block_error(block_message)


class AdultContentBlockAction_LLM(ActionBase):
    """LLM-powered action to block adult content with contextual messaging"""
    
    name = "Adult Content Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for adult content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for adult content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains adult material: {rule_result.details}"
        return self.llm_raise_block_error(reason)








class AdultContentRaiseExceptionAction(ActionBase):
    """Action to raise exception for adult content"""
    
    name = "Adult Content Raise Exception Action"
    description = "Raises DisallowedOperation exception for adult content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for adult content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains adult material or explicit themes "
            "that violate platform policies regarding appropriate content."
        )
        return self.raise_exception(exception_message)


class AdultContentRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for adult content"""
    
    name = "Adult Content Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for adult content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for adult content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains adult material: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Adult Content Block Policy
# Standard policy that blocks adult content using keyword detection
AdultContentBlockPolicy = Policy(
    name="Adult Content Block Policy",
    description="Blocks content containing explicit material and adult themes",
    rule=AdultContentRule(),
    action=AdultContentBlockAction()
)

## Adult Content Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
AdultContentBlockPolicy_LLM = Policy(
    name="Adult Content Block Policy LLM",
    description="Uses LLM to detect and block adult content with contextual understanding",
    rule=AdultContentRule(),
    action=AdultContentBlockAction_LLM()
)

## Adult Content Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
AdultContentBlockPolicy_LLM_Finder = Policy(
    name="Adult Content Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of adult content",
    rule=AdultContentRule_LLM_Finder(),
    action=AdultContentBlockAction()
)





## Adult Content Raise Exception Policy
# Policy that raises exceptions for adult content
AdultContentRaiseExceptionPolicy = Policy(
    name="Adult Content Raise Exception Policy",
    description="Raises DisallowedOperation exception when adult content is detected",
    rule=AdultContentRule(),
    action=AdultContentRaiseExceptionAction()
)

## Adult Content Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
AdultContentRaiseExceptionPolicy_LLM = Policy(
    name="Adult Content Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for adult content",
    rule=AdultContentRule(),
    action=AdultContentRaiseExceptionAction_LLM()
)