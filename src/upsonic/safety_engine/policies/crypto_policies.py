"""
Crypto Currency Block Policy
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class CryptoRule(RuleBase):
    """Rule to detect crypto currency related content"""
    
    name = "Crypto Keywords Rule"
    description = "Detects cryptocurrency keywords and patterns in text"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Default crypto keywords - more specific to avoid false positives
        self.keywords_list = [
            "bitcoin", "btc", "ethereum", "eth", "cryptocurrency", 
            "crypto", "blockchain", "mining", "wallet", "coinbase",
            "binance", "dogecoin", "litecoin", "ripple", "xrp",
            "altcoin", "defi", "nft", "hodl", "satoshi", "hash rate",
            "proof of work", "proof of stake", "smart contract",
            "crypto token", "crypto coin", "crypto exchange", "crypto trading", 
            "crypto pump", "crypto dump", "digital currency", "virtual currency",
        ]
        
        # Context-aware patterns to avoid false positives
        self.false_positive_patterns = [
            r'\bcurrency\s+exchange\b',  # "currency exchange" (foreign exchange)
            r'\bforeign\s+exchange\b',  # "foreign exchange"
            r'\bforex\b',  # "forex"
            r'\bexchange\s+rate\b',  # "exchange rate"
            r'\bexchange\s+office\b',  # "exchange office"
            r'\bexchange\s+bureau\b',  # "exchange bureau"
            r'\bstock\s+exchange\b',  # "stock exchange"
            r'\bcommodity\s+exchange\b',  # "commodity exchange"
            r'\benergy\s+exchange\b',  # "energy exchange"
            r'\bdata\s+exchange\b',  # "data exchange"
            r'\binformation\s+exchange\b',  # "information exchange"
            r'\bknowledge\s+exchange\b',  # "knowledge exchange"
            r'\bstudent\s+exchange\b',  # "student exchange"
            r'\bcultural\s+exchange\b',  # "cultural exchange"
            r'\bexchange\s+program\b',  # "exchange program"
            r'\bexchange\s+student\b',  # "exchange student"
            r'\bexchange\s+visitor\b',  # "exchange visitor"
            r'\bexchange\s+agreement\b',  # "exchange agreement"
            r'\bexchange\s+of\s+ideas\b',  # "exchange of ideas"
            r'\bexchange\s+of\s+information\b',  # "exchange of information"
        ]
        
        # Allow custom keywords from options
        if options and "keywords" in options:
            self.keywords_list.extend(options["keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for crypto content detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or []).lower()
        
        # Find matching keywords with false positive filtering
        triggered_keywords = []
        for keyword in self.keywords_list:
            # Use word boundaries to avoid false positives
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, combined_text):
                # Check for false positive patterns
                is_false_positive = False
                for fp_pattern in self.false_positive_patterns:
                    if re.search(fp_pattern, combined_text.lower()):
                        is_false_positive = True
                        break
                
                if not is_false_positive:
                    triggered_keywords.append(keyword)
        
        # Calculate confidence based on number of matches
        if not triggered_keywords:
                    return RuleOutput(
            confidence=0.0,
            content_type="CRYPTO",
            details="No crypto keywords detected"
        )
        
        return RuleOutput(
            confidence=1.0,
            content_type="CRYPTO",
            details=f"Detected {len(triggered_keywords)} crypto keywords",
            triggered_keywords=triggered_keywords
        )


class CryptoRule_LLM_Finder(RuleBase):
    """Rule to find crypto content using LLM"""
    
    name = "Crypto LLM Finder Rule"
    description = "Finds crypto content using LLM"
    language = "en"  # Default language for this rule
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for crypto content detection"""
        
        triggered_keywords = self._llm_find_keywords_with_input("Crypto", policy_input)
        
        # Calculate confidence based on number of matches
        if not triggered_keywords:
            return RuleOutput(
                confidence=0.0,
                content_type="CRYPTO",
                details="No crypto keywords detected"
            )
        
        return RuleOutput(
            confidence=1.0,
            content_type="CRYPTO",
            details=f"Detected {len(triggered_keywords)} crypto keywords",
            triggered_keywords=triggered_keywords
        )


class CryptoBlockAction(ActionBase):
    """Action for crypto content detection"""
    
    name = "Crypto Keywords Action"
    description = "Handles crypto content with blocking, flagging or transformation"
    language = "en"  # Default language for this action
    
    def __init__(self):
        super().__init__()
        # Crypto-specific messages
        self.error_message = "Cryptocurrency related content detected and blocked."

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute crypto-specific action based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.raise_block_error(message=self.error_message)
        else:
            return self.allow_content()


class CryptoBlockAction_LLM(ActionBase):
    """Action for crypto content detection using LLM"""
    
    name = "Crypto Keywords Action LLM"
    description = "Handles crypto content with blocking, flagging or transformation using LLM"
    language = "en"  # Default language for this action

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute crypto-specific action based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.llm_raise_block_error(reason="Providing investment advice is not a legally permitted operation.")
        else:
            return self.allow_content()


class CryptoReplaceAction(ActionBase):
    """Action for crypto content replacement"""
    
    name = "Crypto Replace Action"
    description = "Replaces crypto keywords with safe alternatives"
    language = "en"  # Default language for this action
    
    def __init__(self):
        super().__init__()
        self.replace_message = "Cryptocurrency related content detected and transformed."
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute crypto-specific action based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.replace_triggered_keywords("NO_CRYPTO_CONTENT")
        else:
            return self.allow_content()


class CryptoRaiseExceptionAction(ActionBase):
    """Action for crypto content that raises exception"""
    
    name = "Crypto Raise Exception Action"
    description = "Raises DisallowedOperation exception when crypto content is detected"
    language = "en"  # Default language for this action
    
    def __init__(self):
        super().__init__()
        self.exception_message = "Cryptocurrency related content detected and operation stopped."

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute crypto-specific action based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.raise_exception(message=self.exception_message)
        else:
            return self.allow_content()


class CryptoRaiseExceptionAction_LLM(ActionBase):
    """Action for crypto content that raises exception using LLM"""
    
    name = "Crypto Raise Exception Action LLM"
    description = "Raises DisallowedOperation exception with LLM-generated message when crypto content is detected"
    language = "en"  # Default language for this action

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute crypto-specific action based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.llm_raise_exception(reason="Providing investment advice is not a legally permitted operation.")
        else:
            return self.allow_content()


# Policy Definitions

## Crypto Block Policy
# Bu policy static şekilde tarama yapar ve eğer kripto keywordleri bulursa direk block aksiyonunu dönderir. 
CryptoBlockPolicy = Policy(
    name="Crypto Block Policy",
    description="Designed for financial institutions to block cryptocurrency discussions",
    rule=CryptoRule(),
    action=CryptoBlockAction(),
)

## Crypto Block Policy LLM Block
# Bu policy static şekilde tarama yapar ve eğer kripto keywordleri bulursa direk block askiyonu dödnerir ama
#  block metinlerini llm ile oluşturur bu sayede generic olur ve aynı openai in son sürümündeki gibi neden engellediğini açıklar.
#  Bu sayede kullanıcıya daha iyi bir deneyim sunarız
CryptoBlockPolicy_LLM_Block = Policy(
    name="Crypto Block Policy LLM Block",
    description="Designed for financial institutions to block cryptocurrency discussions",
    rule=CryptoRule(),
    action=CryptoBlockAction_LLM()
)

## Crypto Block Policy LLM Finder
# Bu policy LLM ile tarama yapar bu sayede daha geniş ölçekli ve iyi bir bulma sistemi çalışmış olur 
# ve eğer kripto keywordleri bulursa direk block askiyonu dödnerir 
CryptoBlockPolicy_LLM_Finder = Policy(
    name="Crypto Block Policy LLM Finder",
    description="Designed for financial institutions to block cryptocurrency discussions",
    rule=CryptoRule_LLM_Finder(),
    action=CryptoBlockAction()
)

# Example of using custom LLM specifications
# CryptoBlockPolicy_Custom_LLMs = Policy(
#     name="Crypto Block Policy with Custom LLMs",
#     description="Example of using custom LLM specifications",
#     rule=CryptoRule_LLM_Finder(),
#     action=CryptoBlockAction_LLM(),
#     language="tr",
#     language_identify_llm=custom_language_llm,  # Custom LLM for language detection
#     base_llm=custom_base_llm,                   # Custom LLM for base operations
#     text_finder_llm=custom_finder_llm           # Custom LLM for text finding
# )

# Example of using model specifications
# CryptoBlockPolicy_Model_Specs = Policy(
#     name="Crypto Block Policy with Model Specs",
#     description="Example of using model specifications",
#     rule=CryptoRule_LLM_Finder(),
#     action=CryptoBlockAction_LLM(),
#     language="auto",
#     language_identify_model="gpt-4",      # Use GPT-4 for language detection
#     base_model="gpt-3.5-turbo",           # Use GPT-3.5 for base operations
#     text_finder_model="gpt-4"             # Use GPT-4 for text finding
# )

## Crypto Replace Policy
# Bu policy static şekilde tarama yapar ve eğer kripto keywordleri bulursa direk replace aksiyonu ile hepsini tek bir static şey ile değiştirir.
CryptoReplace = Policy(
    name="Crypto Replace Policy",
    description="Find and replace crypto keywords with safe alternatives",
    rule=CryptoRule(),
    action=CryptoReplaceAction()
)

CryptoRaiseExceptionPolicy = Policy(
    name="Crypto Raise Exception Policy",
    description="Raises DisallowedOperation exception when crypto content is detected",
    rule=CryptoRule(),
    action=CryptoRaiseExceptionAction()
)

CryptoRaiseExceptionPolicy_LLM_Raise = Policy(
    name="Crypto Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message when crypto content is detected",
    rule=CryptoRule(),
    action=CryptoRaiseExceptionAction_LLM()
)