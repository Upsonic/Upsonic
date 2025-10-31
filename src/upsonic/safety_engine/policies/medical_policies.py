"""
Medical Information Protection Policies (HIPAA Compliance)
Handles detection and protection of medical data, health information, and PHI (Protected Health Information).
"""

import re
from typing import List, Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class MedicalInfoRule(RuleBase):
    """Rule to detect medical information and Protected Health Information (PHI)"""
    
    name = "Medical Information Rule"
    description = "Detects medical records, health information, PHI, and other sensitive medical data"
    language = "en"  # Default language for this rule
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)
        
        # Medical record patterns
        self.medical_record_patterns = [
            # Medical record numbers
            r'\b(?:medical record|patient record|health record|mrn|patient id)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            r'\b(?:case number|case #|file number|file #)\s*:?\s*[A-Z0-9]{4,10}\b',
            # Hospital/Clinic identifiers
            r'\b(?:hospital|clinic|medical center|health center)\s*(?:id|identifier|code)\s*:?\s*[A-Z0-9]{3,8}\b',
        ]
        
        # Health insurance patterns
        self.insurance_patterns = [
            # Health insurance policy numbers
            r'\b(?:health insurance|medical insurance|insurance policy|policy number)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{8,15}\b',
            # Medicare/Medicaid numbers
            r'\b(?:medicare|medicaid)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{8,12}\b',
            # Group numbers
            r'\b(?:group number|group #|group id)\s*:?\s*[A-Z0-9]{4,10}\b',
            # Member ID patterns
            r'\b(?:member id|member number|member #)\s*:?\s*[A-Z0-9]{6,12}\b',
        ]
        
        # Prescription patterns
        self.prescription_patterns = [
            # Prescription numbers
            r'\b(?:prescription|rx|medication)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            # NDC (National Drug Code) patterns
            r'\b(?:ndc|national drug code)\s*(?:number|num|#)?\s*:?\s*\d{4,5}-\d{4,5}-\d{2,3}\b',
            # DEA numbers (for controlled substances)
            r'\b(?:dea number|dea #)\s*:?\s*[A-Z]{2}\d{7}\b',
        ]
        
        # Lab results patterns
        self.lab_patterns = [
            # Lab order numbers
            r'\b(?:lab order|laboratory order|test order)\s*(?:number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            # Lab result identifiers
            r'\b(?:lab result|test result|laboratory result)\s*(?:id|identifier|number|num|#)?\s*:?\s*[A-Z0-9]{6,12}\b',
            # Specimen numbers
            r'\b(?:specimen|sample)\s*(?:number|num|#|id|identifier)?\s*:?\s*[A-Z0-9]{6,12}\b',
        ]
        
        # Medical device patterns
        self.device_patterns = [
            # Medical device serial numbers
            r'\b(?:medical device|device|equipment)\s*(?:serial|model|id|number|num|#)?\s*:?\s*[A-Z0-9]{4,15}\b',
            # Implant device identifiers
            r'\b(?:implant|prosthesis|pacemaker|defibrillator)\s*(?:serial|model|id|number|num|#)?\s*:?\s*[A-Z0-9]{4,15}\b',
        ]
        
        # Medical condition patterns
        self.condition_patterns = [
            # ICD-10 codes
            r'\b[A-Z]\d{2}(?:\.\d{1,3})?\b',
            # ICD-9 codes
            r'\b\d{3}(?:\.\d{1,2})?\b',
            # CPT codes
            r'\b\d{5}(?:-\d{2})?\b',
        ]
        
        # Medical keywords and phrases
        self.medical_keywords = [
            # Patient information
            "patient", "patient name", "patient id", "patient identifier", "patient number", "patient record",
            "medical record", "health record", "clinical record", "patient chart", "medical chart",
            "patient history", "medical history", "health history", "family history",
            
            # Health conditions and diagnoses
            "diagnosis", "diagnoses", "condition", "disease", "illness", "symptom", "symptoms",
            "medical condition", "health condition", "chronic condition", "acute condition",
            "allergy", "allergies", "allergic reaction", "adverse reaction", "side effect",
            # Common medical conditions
            "diabetes", "hypertension", "cancer", "heart disease", "stroke", "asthma", "copd",
            "arthritis", "depression", "anxiety", "migraine", "epilepsy", "alzheimer", "dementia",
            "pneumonia", "bronchitis", "flu", "covid", "coronavirus", "infection", "fever",
            "medical emergency", "emergency room", "er visit", "hospitalization", "admission",
            
            # Medications and treatments
            "medication", "medications", "prescription", "prescriptions", "drug", "drugs",
            "treatment", "treatments", "therapy", "therapies", "procedure", "procedures",
            "surgery", "surgical", "operation", "operative", "anesthesia", "anesthetic",
            "vaccination", "vaccine", "immunization", "injection", "injections",
            
            # Medical professionals and facilities
            "doctor", "physician", "nurse", "nursing", "therapist", "specialist",
            "hospital", "clinic", "medical center", "health center", "urgent care",
            "emergency room", "er", "icu", "intensive care", "surgery center",
            "pharmacy", "pharmacist", "laboratory", "lab", "radiology", "imaging",
            
            # Health insurance and billing
            "health insurance", "medical insurance", "insurance policy", "coverage",
            "copay", "copayment", "deductible", "premium", "claim", "claims",
            "billing", "medical bill", "healthcare cost", "medical expense",
            "medicare", "medicaid", "hmo", "ppo", "health plan",
            
            # Lab results and tests
            "lab results", "laboratory results", "test results", "blood test",
            "urine test", "x-ray", "mri", "ct scan", "ultrasound", "biopsy",
            "pathology", "pathology report", "radiology report", "imaging report",
            
            # Medical devices and equipment
            "medical device", "medical equipment", "prosthesis", "implant",
            "pacemaker", "defibrillator", "hearing aid", "wheelchair", "walker",
            "crutches", "cane", "oxygen", "ventilator", "monitor", "pump",
            
            # Mental health
            "mental health", "psychiatric", "psychology", "psychologist", "psychiatrist",
            "therapy session", "counseling", "depression", "anxiety", "ptsd",
            "bipolar", "schizophrenia", "mental illness", "behavioral health",
            
            # Reproductive health
            "pregnancy", "prenatal", "maternity", "obstetrics", "gynecology",
            "contraception", "birth control", "fertility", "infertility",
            
            # HIPAA and privacy
            "phi", "protected health information", "health information", "medical information",
            "patient privacy", "medical privacy", "health privacy", "confidential",
            "hipaa", "health insurance portability", "accountability act"
        ]
        
        # Sensitive medical information patterns
        self.sensitive_medical_patterns = [
            # HIV/AIDS related
            r'\b(?:hiv|aids|hiv positive|aids positive|hiv negative|aids negative)\b',
            # Mental health conditions
            r'\b(?:suicide|suicidal|self harm|self-harm|depression|anxiety|ptsd|bipolar|schizophrenia)\b',
            # Substance abuse
            r'\b(?:alcoholism|drug addiction|substance abuse|rehabilitation|rehab|detox)\b',
            # Sexual health
            r'\b(?:std|sti|sexually transmitted|pregnancy test|abortion|miscarriage)\b',
        ]
        
        # Allow custom patterns from options
        if options:
            if "custom_patterns" in options:
                self.medical_record_patterns.extend(options["custom_patterns"].get("medical_record", []))
                self.insurance_patterns.extend(options["custom_patterns"].get("insurance", []))
                self.prescription_patterns.extend(options["custom_patterns"].get("prescription", []))
            if "custom_keywords" in options:
                self.medical_keywords.extend(options["custom_keywords"])
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts for medical information detection"""
        
        # Combine all input texts
        combined_text = " ".join(policy_input.input_texts or [])
        
        # Find matching patterns
        triggered_items = []
        
        # Check medical records
        for pattern in self.medical_record_patterns:
            record_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"MEDICAL_RECORD:{record}" for record in record_matches])
        
        # Check insurance information
        for pattern in self.insurance_patterns:
            insurance_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"INSURANCE:{insurance}" for insurance in insurance_matches])
        
        # Check prescriptions
        for pattern in self.prescription_patterns:
            rx_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"PRESCRIPTION:{rx}" for rx in rx_matches])
        
        # Check lab results
        for pattern in self.lab_patterns:
            lab_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"LAB_RESULT:{lab}" for lab in lab_matches])
        
        # Check medical devices
        for pattern in self.device_patterns:
            device_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"MEDICAL_DEVICE:{device}" for device in device_matches])
        
        # Check medical conditions (codes)
        for pattern in self.condition_patterns:
            condition_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"MEDICAL_CODE:{condition}" for condition in condition_matches])
        
        # Check sensitive medical patterns
        for pattern in self.sensitive_medical_patterns:
            sensitive_matches = re.findall(pattern, combined_text, re.IGNORECASE)
            triggered_items.extend([f"SENSITIVE_MEDICAL:{sensitive}" for sensitive in sensitive_matches])
        
        # Check medical keywords with word boundary matching
        for keyword in self.medical_keywords:
            # Use word boundary pattern for better accuracy
            pattern = r'\b' + re.escape(keyword.lower()) + r'(?:s|es|ies|ed|ing)?\b'
            if re.search(pattern, combined_text, re.IGNORECASE):
                triggered_items.append(f"MEDICAL_KEYWORD:{keyword}")
        
        # Calculate confidence based on number and type of matches
        if not triggered_items:
            return RuleOutput(
                confidence=0.0,
                content_type="NO_MEDICAL_INFO",
                details="No medical information detected"
            )
        
        # Weight different types of medical information
        critical_count = len([item for item in triggered_items if any(x in item for x in ["MEDICAL_RECORD:", "INSURANCE:", "SENSITIVE_MEDICAL:"])])
        high_risk_count = len([item for item in triggered_items if any(x in item for x in ["PRESCRIPTION:", "LAB_RESULT:", "MEDICAL_DEVICE:"])])
        medium_risk_count = len([item for item in triggered_items if "MEDICAL_CODE:" in item])
        low_risk_count = len([item for item in triggered_items if "MEDICAL_KEYWORD:" in item])
        
        # Calculate confidence with weighted scoring
        confidence = min(1.0, (critical_count * 1.0 + high_risk_count * 0.8 + medium_risk_count * 0.6 + low_risk_count * 0.3))
        
        return RuleOutput(
            confidence=confidence,
            content_type="MEDICAL_INFO_DETECTED",
            details=f"Detected {len(triggered_items)} medical items: {critical_count} critical, {high_risk_count} high-risk, {medium_risk_count} medium-risk, {low_risk_count} keyword matches",
            triggered_keywords=triggered_items
        )


class MedicalInfoRule_LLM_Finder(RuleBase):
    """LLM-powered rule to detect medical information with better context understanding"""
    
    name = "Medical Info LLM Detection Rule"
    description = "Uses LLM to detect medical information and PHI with context awareness and better accuracy"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process input texts using LLM for medical information detection"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            fallback_rule = MedicalInfoRule()
            return fallback_rule.process(policy_input)
        
        try:
            # Use LLM to find medical information
            triggered_keywords = self._llm_find_keywords_with_input("MEDICAL_INFORMATION", policy_input)
            
            if not triggered_keywords:
                return RuleOutput(
                    confidence=0.0,
                    content_type="NO_MEDICAL_INFO",
                    details="No medical information detected by LLM"
                )
            
            # LLM detection gets high confidence
            confidence = min(1.0, len(triggered_keywords) * 0.8)
            
            return RuleOutput(
                confidence=confidence,
                content_type="MEDICAL_INFO_DETECTED",
                details=f"LLM detected {len(triggered_keywords)} medical information items",
                triggered_keywords=triggered_keywords
            )
            
        except Exception as e:
            # Fallback to pattern-based detection on error
            fallback_rule = MedicalInfoRule()
            return fallback_rule.process(policy_input)


class MedicalInfoBlockAction(ActionBase):
    """Action to block content containing medical information"""
    
    name = "Medical Info Block Action"
    description = "Blocks content containing medical information and PHI"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        block_message = (
            "This content has been blocked as it contains medical information or Protected Health Information (PHI) "
            "that is subject to HIPAA regulations. Please remove any medical information, patient data, "
            "or health records before resubmitting."
        )
        
        return self.raise_block_error(block_message)


class MedicalInfoBlockAction_LLM(ActionBase):
    """LLM-powered action to block medical information content with contextual messaging"""
    
    name = "Medical Info Block Action LLM"
    description = "Uses LLM to generate appropriate block messages for medical information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Use LLM to generate contextual block message
        reason = f"Content contains medical information or PHI: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class MedicalInfoAnonymizeAction(ActionBase):
    """Action to anonymize medical information content"""
    
    name = "Medical Info Anonymize Action"
    description = "Anonymizes medical information while preserving content structure"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute anonymization action for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.anonymize_triggered_keywords()


class MedicalInfoReplaceAction(ActionBase):
    """Action to replace medical information with placeholders"""
    
    name = "Medical Info Replace Action"
    description = "Replaces medical information with safe placeholders"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute replacement action for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        return self.replace_triggered_keywords("[MEDICAL_INFO_REDACTED]")


class MedicalInfoRaiseExceptionAction(ActionBase):
    """Action to raise exception for medical information content"""
    
    name = "Medical Info Raise Exception Action"
    description = "Raises DisallowedOperation exception for medical information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            "DisallowedOperation: Content contains medical information or Protected Health Information (PHI) "
            "that violates HIPAA regulations and medical privacy policies."
        )
        return self.raise_exception(exception_message)


class MedicalInfoRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exception for medical information content"""
    
    name = "Medical Info Raise Exception Action LLM"
    description = "Uses LLM to generate appropriate exception messages for medical information content"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for medical information content"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Content contains medical information or PHI: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Medical Info Block Policy
# Standard policy that blocks content containing medical information using pattern detection
MedicalInfoBlockPolicy = Policy(
    name="Medical Info Block Policy",
    description="Blocks content containing medical information and PHI",
    rule=MedicalInfoRule(),
    action=MedicalInfoBlockAction()
)

## Medical Info Block Policy with LLM
# Enhanced policy using LLM for better context understanding and blocking
MedicalInfoBlockPolicy_LLM = Policy(
    name="Medical Info Block Policy LLM",
    description="Uses LLM to detect and block medical information content with contextual understanding",
    rule=MedicalInfoRule(),
    action=MedicalInfoBlockAction_LLM()
)

## Medical Info Block Policy LLM Finder
# Policy that uses LLM for detection and provides comprehensive blocking
MedicalInfoBlockPolicy_LLM_Finder = Policy(
    name="Medical Info Block Policy LLM Finder",
    description="Uses LLM for both detection and blocking of medical information content",
    rule=MedicalInfoRule_LLM_Finder(),
    action=MedicalInfoBlockAction()
)

## Medical Info Anonymize Policy
# Policy that anonymizes medical information while preserving content structure
MedicalInfoAnonymizePolicy = Policy(
    name="Medical Info Anonymize Policy",
    description="Anonymizes medical information while preserving content",
    rule=MedicalInfoRule(),
    action=MedicalInfoAnonymizeAction()
)

## Medical Info Replace Policy
# Policy that replaces medical information with safe placeholders
MedicalInfoReplacePolicy = Policy(
    name="Medical Info Replace Policy",
    description="Replaces medical information with safe placeholders",
    rule=MedicalInfoRule(),
    action=MedicalInfoReplaceAction()
)

## Medical Info Raise Exception Policy
# Policy that raises exceptions for medical information content
MedicalInfoRaiseExceptionPolicy = Policy(
    name="Medical Info Raise Exception Policy",
    description="Raises DisallowedOperation exception when medical information is detected",
    rule=MedicalInfoRule(),
    action=MedicalInfoRaiseExceptionAction()
)

## Medical Info Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
MedicalInfoRaiseExceptionPolicy_LLM = Policy(
    name="Medical Info Raise Exception Policy LLM",
    description="Raises DisallowedOperation exception with LLM-generated message for medical information content",
    rule=MedicalInfoRule(),
    action=MedicalInfoRaiseExceptionAction_LLM()
)
