"""
Sensitive Social Issues Policies
Handles racism, hate speech, and other discriminatory content.
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class SensitiveSocialRule(RuleBase):
    """Rule to detect sensitive social issues including racism, hate speech, and discriminatory content"""
    
    name = "Sensitive Social Issues Rule"
    description = "Detects racism, hate speech, discriminatory language, and other sensitive social content"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Racial slurs and hate speech keywords (using asterisks to avoid false positives)
        self.hate_speech_keywords = [
            # Racial slurs (partial/coded to avoid false positives)
            "n*gger", "n*gga", "ch*nk", "sp*c", "k*ke", "w*tback", "g*ok", "r*ghead",
            "sand n*gger", "towel head", "camel jockey", "curry muncher", "dot head",
            
            # Religious hate speech
            "christ k*ller", "j*w rat", "muslim terrorist", "islamic extremist",
            "dirty muslim", "terrorist muslim", "bible thumper", "jesus freak",
            
            # LGBTQ+ slurs
            "f*ggot", "f*g", "d*ke", "tr*nny", "sh*male", "he-she", "it",
            
            # Gender-based discrimination
            "stupid woman", "dumb blonde", "kitchen woman", "sandwich maker",
            "gold digger", "slut", "whore", "b*tch", "c*nt", "fuck",
            
            # General discriminatory terms
            "subhuman", "inferior race", "master race", "genetic superiority",
            "breed them out", "ethnic cleansing", "final solution", "white power",
            "black power", "supremacist", "nazi", "hitler was right", "holocaust denial",
            
            # Disability discrimination
            "retard", "r*tard", "mongoloid", "cripple", "gimp", "spastic", "psycho",
            "crazy person", "mental case", "loony", "nutjob"
        ]
        
        # Hate speech patterns
        self.hate_patterns = [
            r'\b(?:all|every|most)\s+(?:blacks?|whites?|jews?|muslims?|christians?|gays?|women|men)\s+(?:are|should)\s+(?:killed?|dead|gone|eliminated)',
            r'\b(?:kill|murder|eliminate|destroy)\s+(?:all|every|most)\s+(?:blacks?|whites?|jews?|muslims?|christians?|gays?|women|men)',
            r'\b(?:blacks?|whites?|jews?|muslims?|christians?|gays?|women|men)\s+(?:are|should be)\s+(?:inferior|superior|slaves?|servants?)',
            r'\b(?:go back to)\s+(?:africa|mexico|china|your country|where you came from)',
            r'\b(?:white|black|jewish|muslim|christian|gay)\s+(?:supremacy|power|pride)\b',
            r'\b(?:race|ethnic|religious)\s+(?:war|cleansing|purity|superiority)\b',
            r'\b(?:burn|lynch|hang|shoot)\s+(?:the|all|every)\s+(?:blacks?|whites?|jews?|muslims?|gays?)',
            r'\b(?:hitler|nazi|kkk|white nationalist)\s+(?:was right|did nothing wrong|had good ideas)'
        ]
        
        # Discriminatory context keywords
        self.discriminatory_keywords = [
            "racial profiling", "systematic racism", "white privilege", "reverse racism",
            "cultural appropriation", "microaggression", "tokenism", "colorblind racism",
            "institutional bias", "prejudice", "bigotry", "xenophobia", "islamophobia",
            "antisemitism", "homophobia", "transphobia", "misogyny", "sexism",
            "ableism", "ageism", "classism", "discrimination", "hate crime",
            "bias incident", "racial slur", "ethnic slur", "derogatory term"
        ]
        
        # Allow custom keywords from options
        if options:
            if "hate_speech_keywords" in options:
                self.hate_speech_keywords.extend(options["hate_speech_keywords"])
            if "hate_patterns" in options:
                self.hate_patterns.extend(options["hate_patterns"])
            if "discriminatory_keywords" in options:
                self.discriminatory_keywords.extend(options["discriminatory_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for sensitive social issues detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or []).lower()
        
        # Find matching hate speech keywords
        triggered_hate_speech = []
        for keyword in self.hate_speech_keywords:
            # Use word boundaries to avoid false positives, handle asterisks and plurals
            clean_keyword = keyword.replace("*", ".")
            pattern = r'\b' + re.escape(keyword.lower()).replace(r'\*', r'[a-z*]') + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text, re.IGNORECASE):
                triggered_hate_speech.append(keyword)
        
        # Find matching hate speech patterns
        triggered_patterns = []
        for pattern in self.hate_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                triggered_patterns.append(pattern)
        
        # Find discriminatory keywords
        triggered_discriminatory = []
        for keyword in self.discriminatory_keywords:
            # Use word boundary pattern that handles plurals and variations
            pattern = r'\b' + re.escape(keyword.lower()) + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text):
                triggered_discriminatory.append(keyword)
        
        # Combine all triggered items
        all_triggered = triggered_hate_speech + triggered_patterns + triggered_discriminatory
        
        # Calculate confidence based on severity and number of matches
        if not all_triggered:
            return RuleOutput(
                confidence=0.0,
                content_type="SAFE_CONTENT",
                details="No sensitive social issues detected"
            )
        
        # Higher confidence for hate speech, moderate for discriminatory content
        hate_speech_weight = len(triggered_hate_speech) * 0.9
        pattern_weight = len(triggered_patterns) * 0.95
        discriminatory_weight = len(triggered_discriminatory) * 0.3
        
        total_weight = hate_speech_weight + pattern_weight + discriminatory_weight
        confidence = min(1.0, total_weight)
        
        return RuleOutput(
            confidence=confidence,
            content_type="SENSITIVE_SOCIAL_CONTENT",
            details=f"Detected {len(all_triggered)} sensitive social issues: {len(triggered_hate_speech)} hate speech terms, {len(triggered_patterns)} hate patterns, {len(triggered_discriminatory)} discriminatory terms",
            triggered_keywords=all_triggered
        )


class SensitiveSocialRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect sensitive social issues with better context understanding"""
    
    name = "Sensitive Social Issues LLM Rule"
    description = "Uses LLM to detect racism, hate speech, and discriminatory content with context awareness"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for sensitive social issues detection"""
        
        if not self.text_finder_llm:
            # Fallback to keyword-based detection
            fallback_rule = SensitiveSocialRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find sensitive social content
            triggered_keywords = self._llm_find_keywords_with_input("SENSITIVE_SOCIAL_CONTENT", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="SAFE_CONTENT",
                    details="No sensitive social issues detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="SENSITIVE_SOCIAL_CONTENT",
                details=f"LLM detected {len(triggered_keywords)} sensitive social issues",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to keyword-based detection on error
            fallback_rule = SensitiveSocialRule()
            return fallback_rule.process(policy_input)


class SensitiveSocialBlockAction(ActionBase):
    """Action to block sensitive social content with appropriate messaging"""
    
    name = "Sensitive Social Block Action"
    description = "Blocks content containing sensitive social issues with educational message"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for sensitive social content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Block with educational message
        block_message = (
            "This content has been blocked as it contains language or themes related to "
            "discrimination, hate speech, or sensitive social issues. Please ensure your "
            "communication is respectful and inclusive of all individuals regardless of "
            "race, religion, gender, sexual orientation, or other personal characteristics."
        )
        
        return self.raise_block_error(block_message)


class SensitiveSocialBlockAction_LLM(ActionBase):
    """LLM-powered action to block sensitive social content with contextual messaging"""
    
    name = "Sensitive Social Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for sensitive social content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for sensitive social content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains sensitive social issues: {rule_result.details}"
        return self.llm_raise_block_error(reason)








class SensitiveSocialRaiseExceptionAction(ActionBase):
    """Action to raise exception for sensitive social content"""
    
    name = "Sensitive Social Raise Exception Action"
    description = "Raises DisallowedOperation exception for sensitive social content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for sensitive social content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains sensitive social issues including "
            "discrimination, hate speech, or offensive language that violates platform policies."
        )
        return self.raise_exception(exception_message)


class SensitiveSocialRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for sensitive social content"""
    
    name = "Sensitive Social Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for sensitive social content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for sensitive social content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains sensitive social issues: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Sensitive Social Block Policy
# Standard policy that blocks content with sensitive social issues using keyword detection
SensitiveSocialBlockPolicy = Policy(
    name="Sensitive Social Block Policy",
    description="Blocks content containing racism, hate speech, and discriminatory language",
    rule=SensitiveSocialRule(),
    action=SensitiveSocialBlockAction()
)

## Sensitive Social Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
SensitiveSocialBlockPolicy_LLM = Policy(
    name="Sensitive Social Block Policy LLM",
    description="Uses LLM to detect and block sensitive social content with contextual understanding",
    rule=SensitiveSocialRule(),
    action=SensitiveSocialBlockAction_LLM()
)

## Sensitive Social Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
SensitiveSocialBlockPolicy_LLM_Finder = Policy(
    name="Sensitive Social Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of sensitive social content",
    rule=SensitiveSocialRule_LLM_Finder(),
    action=SensitiveSocialBlockAction()
)





## Sensitive Social Raise Exception Policy
# Policy that raises exceptions for sensitive social content
SensitiveSocialRaiseExceptionPolicy = Policy(
    name="Sensitive Social Raise Exception Policy",
    description="Raises DisallowedOperation exception when sensitive social content is detected",
    rule=SensitiveSocialRule(),
    action=SensitiveSocialRaiseExceptionAction()
)

## Sensitive Social Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
SensitiveSocialRaiseExceptionPolicy_LLM = Policy(
    name="Sensitive Social Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for sensitive social content",
    rule=SensitiveSocialRule(),
    action=SensitiveSocialRaiseExceptionAction_LLM()
)