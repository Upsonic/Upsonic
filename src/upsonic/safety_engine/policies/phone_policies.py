"""
Phone Number Anonymization Policies
"""

import re
from typing import List
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class AnonymizePhoneNumberRule(RuleBase):
    """Rule to anonymize phone numbers"""
    
    name = "Anonymize Number Rule"
    description = "Anonymize phone numbers"
    language = "en"  # Default language for this rule
    
    def __init__(self):
        super().__init__()
        # Updated regex to match various phone number formats (Turkish and international)
        self.pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for phone number anonymization"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or []).lower()
        
        # Find matching phone numbers
        triggered_phone_numbers = []
        for match in re.finditer(self.pattern, combined_text):
            triggered_phone_numbers.append(match.group(0))
        
        # Calculate confidence based on number of matches
        if not triggered_phone_numbers:
            return RuleOutput(
                confidence=0.0,
                content_type="PHONE_NUMBER",
                details="No phone numbers detected"
            )
        
        return RuleOutput(
            confidence=1.0,
            content_type="PHONE_NUMBER",
            details=f"Detected {len(triggered_phone_numbers)} phone numbers",
            triggered_keywords=triggered_phone_numbers
        )


class AnonymizePhoneNumberRule_LLM_Finder(RuleBase):
    """Rule to anonymize phone numbers using LLM"""
    
    name = "Anonymize Number Rule LLM"
    description = "Anonymize phone numbers using LLM"
    language = "en"  # Default language for this rule

    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for phone number anonymization"""
        
        triggered_keywords = self._llm_find_keywords_with_input("Phone Number", policy_input)

        if not triggered_keywords:
            return RuleOutput(
                confidence=0.0,
                content_type="PHONE_NUMBER",
                details="No phone numbers detected"
            )
        
        return RuleOutput(
            confidence=1.0,
            content_type="PHONE_NUMBER",
            details=f"Detected {len(triggered_keywords)} phone numbers",
            triggered_keywords=triggered_keywords
        )


class AnonymizePhoneNumberAction(ActionBase):
    """Action for phone number anonymization"""
    
    name = "Anonymize Phone Number Action"
    description = "Anonymizes phone numbers"
    language = "en"  # Default language for this action

    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute phone number anonymization based on rule confidence"""
        if rule_result.confidence >= 0.8:
            return self.anonymize_triggered_keywords()
        else:
            return self.allow_content()



# Policy Definitions

## Anonymize Phone Numbers Policy
# Bu policy static şekilde tarama yapar ve eğer telefon numarası bulursa direk anonymize aksiyonu çalışır. 
# Bu da sayıları başka random sayılarla harfleri ise başka random harflerle değiştirir.
AnonymizePhoneNumbersPolicy = Policy(
    name="Anonymize Phone Numbers Policy",
    description="Anonymize phone numbers",
    rule=AnonymizePhoneNumberRule(),
    action=AnonymizePhoneNumberAction()
)



## Anonymize Phone Numbers Policy LLM Finder + LLM Anonymize
# Bu policy LLM ile tarama yapar bu sayede daha geniş ölçekli ve iyi bir bulma sistemi çalışmış olur 
# ve eğer telefon numarası bulursa direk anonymize aksiyonu çalışır ama bu fonksiyonun llm versiyonunu kullanır bu sayede 
# daha kapsayıcı ve daha iyi bir anonymize işlemi yapılır.
AnonymizePhoneNumbersPolicy_LLM_Finder = Policy(
    name="Anonymize Phone Numbers Policy LLM Finder + Anonymize",
    description="Anonymize phone numbers using LLM for both finding and anonymization",
    rule=AnonymizePhoneNumberRule_LLM_Finder(),
    action=AnonymizePhoneNumberAction()
)