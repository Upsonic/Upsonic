"""
Upsonic LLM Provider for AI Safety Engine
"""

from typing import List, Optional, Union
import asyncio
from pydantic import BaseModel
from upsonic.models import Model

from upsonic.tasks.tasks import Task

class KeywordDetectionResponse(BaseModel):
    """Response format for keyword detection"""
    detected_keywords: List[str]
    confidence: float
    reasoning: str


class BlockMessageResponse(BaseModel):
    """Response format for block message generation"""
    block_message: str
    severity: str
    reasoning: str


class AnonymizationResponse(BaseModel):
    """Response format for content anonymization"""
    anonymized_content: str
    anonymized_parts: List[str]
    reasoning: str


class LanguageDetectionResponse(BaseModel):
    """Response format for language detection"""
    language_code: str  # ISO 639-1 code (e.g., 'tr', 'en', 'es', 'fr')
    language_name: str  # Full language name (e.g., 'Turkish', 'English')
    confidence: float   # Confidence score 0.0-1.0


class TranslationResponse(BaseModel):
    """Response format for text translation"""
    translated_text: str
    source_language: str
    target_language: str
    confidence: float


class ToolSafetyAnalysisResponse(BaseModel):
    """Response format for tool safety analysis"""
    is_harmful: bool = False
    is_malicious: bool = False
    confidence: float
    reasons: List[str]
    threat_categories: List[str] = []
    suspicious_args: List[str] = []
    recommendation: str

class PolicyFeedbackResponse(BaseModel):
    """Response format for policy feedback generation.
    
    Used when a policy violation occurs and feedback_enabled is True.
    Generates constructive feedback to help users/agents understand
    what violated the policy and how to correct it.
    """
    feedback_message: str  # The constructive feedback message for the user/agent
    suggested_approach: str  # Suggestion for how to proceed or rephrase
    violation_type: str  # Type of violation detected (e.g., "CRYPTO", "PII", "ADULT_CONTENT")
    severity: str  # "warning", "moderate", "critical"



class UpsonicLLMProvider:
    """Upsonic-based LLM provider for AI Safety Engine"""
    
    def __init__(self, agent_name: str = "AI Safety Agent", model: Union[Model, str] = None):
        from upsonic.agent.agent import Agent
        if model:
            self.agent = Agent(model=model, name=agent_name)
        else:
            self.agent = Agent(name=agent_name)
    
    def find_keywords(self, content_type: str, text: str, language: str = "en") -> List[str]:
        """Find keywords of specified content type in text using Upsonic Agent"""
        
        language_instruction = ""
        if language == "tr":
            language_instruction = " Analyze the text in Turkish context."
        elif language == "en":
            language_instruction = " Analyze the text in English context."
        else:
            language_instruction = f" Analyze the text in {language} language context."
        
        # Generalist prompt: For sensitive content types, extract only explicit instances (e.g., phone numbers, emails, crypto addresses) and ignore context words unless the content type is inherently contextual.
        detection_instruction = (
            f"Detect and extract only explicit instances of {content_type.lower()} from the text (such as actual numbers, addresses, or identifiers, proper noun, things). "
            "Do NOT include general context words, phrases, or references unless the content type requires them (e.g., for 'topic' or 'intent'). "
            "Return only the actual detected items as keywords."
        )
        
        task = Task(
            f"{detection_instruction}{language_instruction}\n\nText: {text}",
            response_format=KeywordDetectionResponse
        )
        
        try:
            result = self.agent.do(task)
            
            if result.confidence >= 0.7:  # High confidence threshold
                return result.detected_keywords
            else:
                return []
                
        except Exception as e:
            return []

    async def find_keywords_async(self, content_type: str, text: str, language: str = "en") -> List[str]:
        """Async keyword detection using Upsonic Agent.do_async."""
        language_instruction = ""
        if language == "tr":
            language_instruction = " Analyze the text in Turkish context."
        elif language == "en":
            language_instruction = " Analyze the text in English context."
        else:
            language_instruction = f" Analyze the text in {language} language context."

        detection_instruction = (
            f"Detect and extract only explicit instances of {content_type.lower()} from the text (such as actual numbers, addresses, or identifiers, proper noun, things). "
            "Do NOT include general context words, phrases, or references unless the content type requires them (e.g., for 'topic' or 'intent'). "
            "Return only the actual detected items as keywords."
        )

        task = Task(
            f"{detection_instruction}{language_instruction}\n\nText: {text}",
            response_format=KeywordDetectionResponse
        )

        try:
            result = await self.agent.do_async(task)
            if result.confidence >= 0.7:
                return result.detected_keywords
            return []
        except Exception:
            return []
    
    def generate_block_message(self, reason: str, language: str = "en") -> str:
        """Generate contextual block message using Upsonic Agent"""
        
        language_instruction = ""
        if language == "tr":
            language_instruction = " Respond in Turkish."
        elif language == "en":
            language_instruction = " Respond in English."
        else:
            language_instruction = f" Respond in the language with code '{language}'."
        
        task = Task(
            f"Generate a professional and clear block message for the following reason: {reason}. "
            f"The message should be informative but not harsh, explaining why the content was blocked. First tell the what happened and then explain why it was blocked."
            f"Keep it concise and user-friendly.{language_instruction}",
            response_format=BlockMessageResponse
        )
        
        try:
            result = self.agent.do(task)
            return result.block_message
            
        except Exception as e:
            return f"Content blocked: {reason}"

    async def generate_block_message_async(self, reason: str, language: str = "en") -> str:
        """Async block message generation using Upsonic Agent.do_async."""
        language_instruction = ""
        if language == "tr":
            language_instruction = " Respond in Turkish."
        elif language == "en":
            language_instruction = " Respond in English."
        else:
            language_instruction = f" Respond in the language with code '{language}'."

        task = Task(
            f"Generate a professional and clear block message for the following reason: {reason}. "
            f"The message should be informative but not harsh, explaining why the content was blocked. First tell the what happened and then explain why it was blocked."
            f"Keep it concise and user-friendly.{language_instruction}",
            response_format=BlockMessageResponse
        )
        try:
            result = await self.agent.do_async(task)
            return result.block_message
        except Exception:
            return f"Content blocked: {reason}"
    
    def anonymize_content(self, text: str, keywords: List[str], language: str = "en") -> str:
        """Anonymize content by replacing sensitive keywords using Upsonic Agent"""
        
        if not keywords:
            return text
        
        language_instruction = ""
        if language == "tr":
            language_instruction = " Maintain Turkish language structure and context."
        elif language == "en":
            language_instruction = " Maintain English language structure and context."
        else:
            language_instruction = f" Maintain {language} language structure and context."
        
        task = Task(
            f"Anonymize the following text by replacing these sensitive keywords: {keywords}. "
            f"Replace them with appropriate placeholders while maintaining readability and context. "
            f"Be careful to only replace the exact keywords provided.{language_instruction}\n\nText: {text}",
            response_format=AnonymizationResponse
        )
        
        try:
            result = self.agent.do(task)
            return result.anonymized_content
            
        except Exception as e:
            # Fallback to simple replacement
            anonymized = text
            for keyword in keywords:
                anonymized = anonymized.replace(keyword, "[REDACTED]")
            return anonymized

    async def anonymize_content_async(self, text: str, keywords: List[str], language: str = "en") -> str:
        """Async anonymization using Upsonic Agent.do_async with graceful fallback."""
        if not keywords:
            return text

        language_instruction = ""
        if language == "tr":
            language_instruction = " Maintain Turkish language structure and context."
        elif language == "en":
            language_instruction = " Maintain English language structure and context."
        else:
            language_instruction = f" Maintain {language} language structure and context."

        task = Task(
            f"Anonymize the following text by replacing these sensitive keywords: {keywords}. "
            f"Replace them with appropriate placeholders while maintaining readability and context. "
            f"Be careful to only replace the exact keywords provided.{language_instruction}\n\nText: {text}",
            response_format=AnonymizationResponse
        )
        try:
            result = await self.agent.do_async(task)
            return result.anonymized_content
        except Exception:
            anonymized = text
            for keyword in keywords:
                anonymized = anonymized.replace(keyword, "[REDACTED]")
            return anonymized
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the given text using Upsonic Agent"""
        
        if not text or not text.strip():
            return "en"  # Default to English for empty text
        
        task = Task(
            f"Detect the language of the following text and return the ISO 639-1 language code (e.g., 'en', 'tr', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'). "
            f"Analyze the text carefully and provide the most likely language using the standard ISO 639-1 two-letter language codes.\n\nText: {text}",
            response_format=LanguageDetectionResponse
        )
        
        try:
            result = self.agent.do(task)
            
            if result.confidence >= 0.6:  # Reasonable confidence threshold
                return result.language_code
            else:
                return "en"  # Default to English if confidence is low
                
        except Exception as e:
            return "en"  # Default fallback

    async def detect_language_async(self, text: str) -> str:
        """Async language detection using Upsonic Agent.do_async."""
        if not text or not text.strip():
            return "en"
        task = Task(
            f"Detect the language of the following text and return the ISO 639-1 language code (e.g., 'en', 'tr', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'). "
            f"Analyze the text carefully and provide the most likely language using the standard ISO 639-1 two-letter language codes.\n\nText: {text}",
            response_format=LanguageDetectionResponse
        )
        try:
            result = await self.agent.do_async(task)
            if result.confidence >= 0.6:
                return result.language_code
            return "en"
        except Exception:
            return "en"
    
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using Upsonic Agent"""
        
        if not text or not text.strip():
            return text
        
        # Map ISO 639-1 language codes to full names for better translation
        language_map = {
            # Major languages
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "tr": "Turkish",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "nl": "Dutch",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "cs": "Czech",
            "sk": "Slovak",
            "hu": "Hungarian",
            "ro": "Romanian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sr": "Serbian",
            "sl": "Slovenian",
            "et": "Estonian",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "mt": "Maltese",
            "ga": "Irish",
            "cy": "Welsh",
            "is": "Icelandic",
            "fo": "Faroese",
            "mk": "Macedonian",
            "sq": "Albanian",
            "bs": "Bosnian",
            "me": "Montenegrin",
            "uk": "Ukrainian",
            "be": "Belarusian",
            "kk": "Kazakh",
            "ky": "Kyrgyz",
            "uz": "Uzbek",
            "tg": "Tajik",
            "mn": "Mongolian",
            "ka": "Georgian",
            "hy": "Armenian",
            "az": "Azerbaijani",
            "fa": "Persian",
            "ur": "Urdu",
            "pa": "Punjabi",
            "gu": "Gujarati",
            "or": "Odia",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam",
            "si": "Sinhala",
            "my": "Burmese",
            "km": "Khmer",
            "lo": "Lao",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Tagalog",
            "he": "Hebrew",
            "yi": "Yiddish",
            "am": "Amharic",
            "sw": "Swahili",
            "zu": "Zulu",
            "af": "Afrikaans",
            "xh": "Xhosa",
            "st": "Southern Sotho",
            "tn": "Tswana",
            "ts": "Tsonga",
            "ss": "Swati",
            "ve": "Venda",
            "nr": "Southern Ndebele",
            "nd": "Northern Ndebele"
        }
        
        target_lang_name = language_map.get(target_language, target_language)
        
        task = Task(
            f"System: You are a professional translator. Your task is to translate the following text from English to {target_lang_name}.\n\n"
            f"Rules:\n"
            f"1. ALWAYS translate to {target_lang_name}, never return the original text\n"
            f"2. Maintain technical terms but translate surrounding context\n"
            f"3. Ensure natural flow in {target_lang_name}\n"
            f"4. Keep the same tone and formality level\n"
            f"5. If you see 'cryptocurrency' or 'crypto', translate it as 'kripto para'\n\n"
            f"Text to translate:\n{text}\n\n"
            f"Important: Return ONLY the translation in {target_lang_name}. Do not include any explanations or the original text.",
            response_format=TranslationResponse
        )
        
        try:
            result = self.agent.do(task)
            
            translated = result.translated_text.strip()
            
            # If translation is empty or exactly the same as input, try one more time
            if not translated or translated == text.strip():
                task.description += "\n\nWARNING: Previous attempt returned original text. Please ensure to translate to " + target_lang_name
                result = self.agent.do(task)
                translated = result.translated_text.strip()
            
            # Final check - don't return original text
            if translated and translated != text.strip():
                return translated
            else:
                # Emergency fallback translations for common messages
                fallback_translations = {
                    "Cryptocurrency related content detected and blocked.": "Kripto para ile ilgili içerik tespit edildi ve engellendi.",
                    "Content blocked:": "İçerik engellendi:",
                }
                
                for eng, tr in fallback_translations.items():
                    if text.strip() == eng:
                        return tr
                
                return text
            
        except Exception as e:
            return text  # Fallback to original text

    async def translate_text_async(self, text: str, target_language: str) -> str:
        """Async translation using Upsonic Agent.do_async with safeguards."""
        if not text or not text.strip():
            return text

        language_map = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "tr": "Turkish",
            "ar": "Arabic",
            "hi": "Hindi",
            "bn": "Bengali",
            "nl": "Dutch",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "cs": "Czech",
            "sk": "Slovak",
            "hu": "Hungarian",
            "ro": "Romanian",
            "bg": "Bulgarian",
            "hr": "Croatian",
            "sr": "Serbian",
            "sl": "Slovenian",
            "et": "Estonian",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "mt": "Maltese",
            "ga": "Irish",
            "cy": "Welsh",
            "is": "Icelandic",
            "fo": "Faroese",
            "mk": "Macedonian",
            "sq": "Albanian",
            "bs": "Bosnian",
            "me": "Montenegrin",
            "uk": "Ukrainian",
            "be": "Belarusian",
            "kk": "Kazakh",
            "ky": "Kyrgyz",
            "uz": "Uzbek",
            "tg": "Tajik",
            "mn": "Mongolian",
            "ka": "Georgian",
            "hy": "Armenian",
            "az": "Azerbaijani",
            "fa": "Persian",
            "ur": "Urdu",
            "pa": "Punjabi",
            "gu": "Gujarati",
            "or": "Odia",
            "ta": "Tamil",
            "te": "Telugu",
            "kn": "Kannada",
            "ml": "Malayalam",
            "si": "Sinhala",
            "my": "Burmese",
            "km": "Khmer",
            "lo": "Lao",
            "th": "Thai",
            "vi": "Vietnamese",
            "id": "Indonesian",
            "ms": "Malay",
            "tl": "Tagalog",
            "he": "Hebrew",
            "yi": "Yiddish",
            "am": "Amharic",
            "sw": "Swahili",
            "zu": "Zulu",
            "af": "Afrikaans",
            "xh": "Xhosa",
            "st": "Southern Sotho",
            "tn": "Tswana",
            "ts": "Tsonga",
            "ss": "Swati",
            "ve": "Venda",
            "nr": "Southern Ndebele",
            "nd": "Northern Ndebele"
        }

        target_lang_name = language_map.get(target_language, target_language)
        task = Task(
            f"System: You are a professional translator. Your task is to translate the following text from English to {target_lang_name}.\n\n"
            f"Rules:\n"
            f"1. ALWAYS translate to {target_lang_name}, never return the original text\n"
            f"2. Maintain technical terms but translate surrounding context\n"
            f"3. Ensure natural flow in {target_lang_name}\n"
            f"4. Keep the same tone and formality level\n"
            f"5. If you see 'cryptocurrency' or 'crypto', translate it as 'kripto para'\n\n"
            f"Text to translate:\n{text}\n\n"
            f"Important: Return ONLY the translation in {target_lang_name}. Do not include any explanations or the original text.",
            response_format=TranslationResponse
        )
        try:
            result = await self.agent.do_async(task)
            translated = result.translated_text.strip()
            if not translated or translated == text.strip():
                task.description += "\n\nWARNING: Previous attempt returned original text. Please ensure to translate to " + target_lang_name
                result = await self.agent.do_async(task)
                translated = result.translated_text.strip()
            if translated and translated != text.strip():
                return translated
            fallback_translations = {
                "Cryptocurrency related content detected and blocked.": "Kripto para ile ilgili içerik tespit edildi ve engellendi.",
                "Content blocked:": "İçerik engellendi:",
            }
            for eng, tr in fallback_translations.items():
                if text.strip() == eng:
                    return tr
            return text
        except Exception:
            return text
    
    def analyze_tool_safety(
        self,
        tool_info: dict,
        analysis_type: str = "HARMFUL_FUNCTIONALITY"
    ) -> dict:
        """
        Analyze tool safety using LLM.
        
        Args:
            tool_info: Dictionary containing tool information (name, description, parameters, arguments)
            analysis_type: Type of analysis - "HARMFUL_FUNCTIONALITY" or "MALICIOUS_CALL"
        
        Returns:
            Dictionary with analysis results
        """
        import json
        
        tool_name = tool_info.get("name", "unknown")
        tool_description = tool_info.get("description", "")
        tool_parameters = tool_info.get("parameters", {})
        tool_arguments = tool_info.get("arguments", {})
        
        if analysis_type == "HARMFUL_FUNCTIONALITY":
            # Analyze tool for harmful functionality during registration
            prompt = (
                f"Analyze this tool for harmful or dangerous functionality:\n\n"
                f"Tool Name: {tool_name}\n"
                f"Description: {tool_description}\n"
                f"Parameters Schema: {json.dumps(tool_parameters, indent=2)}\n\n"
                f"Determine if this tool could be used for:\n"
                f"1. System manipulation (shutdown, restart, privilege escalation)\n"
                f"2. Data destruction (file deletion, database drops, data wiping)\n"
                f"3. Network attacks (port scanning, DoS, intrusion)\n"
                f"4. Security violations (credential theft, encryption bypass)\n"
                f"5. Malicious operations (malware installation, backdoors)\n\n"
                f"Provide your analysis with confidence score (0.0-1.0), specific reasons, "
                f"and threat categories if harmful."
            )
            
            task = Task(prompt, response_format=ToolSafetyAnalysisResponse)
            
            try:
                result = self.agent.do(task)
                return {
                    "is_harmful": result.is_harmful,
                    "confidence": result.confidence,
                    "reasons": result.reasons,
                    "threat_categories": result.threat_categories,
                    "recommendation": result.recommendation
                }
            except Exception as e:
                return {
                    "is_harmful": False,
                    "confidence": 0.0,
                    "reasons": ["Analysis failed"],
                    "threat_categories": []
                }
        
        elif analysis_type == "MALICIOUS_CALL":
            # Analyze specific tool call for malicious arguments
            prompt = (
                f"Analyze this tool call for malicious or suspicious arguments:\n\n"
                f"Tool Name: {tool_name}\n"
                f"Description: {tool_description}\n"
                f"Arguments Provided: {json.dumps(tool_arguments, indent=2)}\n"
                f"Parameters Schema: {json.dumps(tool_parameters, indent=2)}\n\n"
                f"Check for:\n"
                f"1. Dangerous file paths (system directories, sensitive files)\n"
                f"2. Suspicious commands or scripts (rm -rf, dd, format)\n"
                f"3. Command injection attempts (&&, ||, ;, |, `)\n"
                f"4. Unusual parameter combinations indicating attacks\n"
                f"5. Attempts to access unauthorized resources\n\n"
                f"Provide analysis with confidence score, reasons, and list suspicious arguments."
            )
            
            task = Task(prompt, response_format=ToolSafetyAnalysisResponse)
            
            try:
                result = self.agent.do(task)
                return {
                    "is_malicious": result.is_malicious,
                    "confidence": result.confidence,
                    "reasons": result.reasons,
                    "suspicious_args": result.suspicious_args,
                    "recommendation": result.recommendation
                }
            except Exception as e:
                return {
                    "is_malicious": False,
                    "confidence": 0.0,
                    "reasons": ["Analysis failed"],
                    "suspicious_args": []
                }
        
        else:
            return {"error": "Unknown analysis type"}
    
    async def analyze_tool_safety_async(
        self,
        tool_info: dict,
        analysis_type: str = "HARMFUL_FUNCTIONALITY"
    ) -> dict:
        """
        Async version of analyze_tool_safety.
        """
        import json
        
        tool_name = tool_info.get("name", "unknown")
        tool_description = tool_info.get("description", "")
        tool_parameters = tool_info.get("parameters", {})
        tool_arguments = tool_info.get("arguments", {})
        
        if analysis_type == "HARMFUL_FUNCTIONALITY":
            prompt = (
                f"Analyze this tool for harmful or dangerous functionality:\n\n"
                f"Tool Name: {tool_name}\n"
                f"Description: {tool_description}\n"
                f"Parameters Schema: {json.dumps(tool_parameters, indent=2)}\n\n"
                f"Determine if this tool could be used for:\n"
                f"1. System manipulation (shutdown, restart, privilege escalation)\n"
                f"2. Data destruction (file deletion, database drops, data wiping)\n"
                f"3. Network attacks (port scanning, DoS, intrusion)\n"
                f"4. Security violations (credential theft, encryption bypass)\n"
                f"5. Malicious operations (malware installation, backdoors)\n\n"
                f"Provide your analysis with confidence score (0.0-1.0), specific reasons, "
                f"and threat categories if harmful."
            )
            
            task = Task(prompt, response_format=ToolSafetyAnalysisResponse)
            
            try:
                result = await self.agent.do_async(task)
                return {
                    "is_harmful": result.is_harmful,
                    "confidence": result.confidence,
                    "reasons": result.reasons,
                    "threat_categories": result.threat_categories,
                    "recommendation": result.recommendation
                }
            except Exception:
                return {
                    "is_harmful": False,
                    "confidence": 0.0,
                    "reasons": ["Analysis failed"],
                    "threat_categories": []
                }
        
        elif analysis_type == "MALICIOUS_CALL":
            prompt = (
                f"Analyze this tool call for malicious or suspicious arguments:\n\n"
                f"Tool Name: {tool_name}\n"
                f"Description: {tool_description}\n"
                f"Arguments Provided: {json.dumps(tool_arguments, indent=2)}\n"
                f"Parameters Schema: {json.dumps(tool_parameters, indent=2)}\n\n"
                f"Check for:\n"
                f"1. Dangerous file paths (system directories, sensitive files)\n"
                f"2. Suspicious commands or scripts (rm -rf, dd, format)\n"
                f"3. Command injection attempts (&&, ||, ;, |, `)\n"
                f"4. Unusual parameter combinations indicating attacks\n"
                f"5. Attempts to access unauthorized resources\n\n"
                f"Provide analysis with confidence score, reasons, and list suspicious arguments."
            )
            
            task = Task(prompt, response_format=ToolSafetyAnalysisResponse)
            
            try:
                result = await self.agent.do_async(task)
                return {
                    "is_malicious": result.is_malicious,
                    "confidence": result.confidence,
                    "reasons": result.reasons,
                    "suspicious_args": result.suspicious_args,
                    "recommendation": result.recommendation
                }
            except Exception:
                return {
                    "is_malicious": False,
                    "confidence": 0.0,
                    "reasons": ["Analysis failed"],
                    "suspicious_args": []
                }
        
        else:
            return {"error": "Unknown analysis type"}

    def generate_policy_feedback(
        self,
        original_content: str,
        policy_name: str,
        violation_reason: str,
        policy_type: str,
        action_type: str,
        language: str = "en"
    ) -> str:
        """
        Generate constructive feedback for a policy violation.
        
        This method creates helpful, user-friendly feedback explaining what
        violated the policy and how to correct it.
        
        Args:
            original_content: The content that violated the policy
            policy_name: Name of the violated policy
            violation_reason: Details about why the policy was violated
            policy_type: Either "user_policy" or "agent_policy"
            action_type: The action that would be taken (BLOCK, REPLACE, ANONYMIZE, RAISE)
            language: ISO 639-1 language code for the feedback
            
        Returns:
            str: Constructive feedback message
        """
        language_instruction = ""
        if language == "tr":
            language_instruction = " Respond in Turkish."
        elif language == "en":
            language_instruction = " Respond in English."
        else:
            language_instruction = f" Respond in the language with code '{language}'."
        
        if policy_type == "user_policy":
            context = (
                "You are providing feedback to a user whose input violated a safety policy. "
                "Be helpful and constructive. Explain what was wrong and suggest how they can "
                "rephrase or modify their request to comply with the policy."
            )
        else:  # agent_policy
            context = (
                "You are providing feedback to an AI agent whose response violated a safety policy. "
                "The agent will receive this feedback and try again. Be clear about what was wrong "
                "and provide specific guidance on how to generate a compliant response."
            )
        
        # Truncate content if too long
        content_preview = original_content[:500] + "..." if len(original_content) > 500 else original_content
        
        prompt = (
            f"{context}{language_instruction}\n\n"
            f"Policy Name: {policy_name}\n"
            f"Violation Reason: {violation_reason}\n"
            f"Action That Would Be Taken: {action_type}\n"
            f"Content Preview: {content_preview}\n\n"
            f"Generate constructive feedback that:\n"
            f"1. Explains clearly what aspect of the content violated the policy\n"
            f"2. Provides specific suggestions for how to modify the content\n"
            f"3. Is professional and helpful, not harsh or accusatory\n"
            f"4. Does NOT include the original problematic content in your response"
        )
        
        task = Task(prompt, response_format=PolicyFeedbackResponse)
        
        try:
            result = self.agent.do(task)
            # Combine feedback message with suggested approach for comprehensive feedback
            return f"{result.feedback_message}\n\nSuggested approach: {result.suggested_approach}"
        except Exception as e:
            # Fallback to generic feedback message
            return (
                f"Your content was flagged by the '{policy_name}' policy. "
                f"Reason: {violation_reason}. Please modify your content to comply with this policy."
            )
    
    async def generate_policy_feedback_async(
        self,
        original_content: str,
        policy_name: str,
        violation_reason: str,
        policy_type: str,
        action_type: str,
        language: str = "en"
    ) -> str:
        """
        Async version of generate_policy_feedback.
        
        Generate constructive feedback for a policy violation asynchronously.
        
        Args:
            original_content: The content that violated the policy
            policy_name: Name of the violated policy
            violation_reason: Details about why the policy was violated
            policy_type: Either "user_policy" or "agent_policy"
            action_type: The action that would be taken (BLOCK, REPLACE, ANONYMIZE, RAISE)
            language: ISO 639-1 language code for the feedback
            
        Returns:
            str: Constructive feedback message
        """
        language_instruction = ""
        if language == "tr":
            language_instruction = " Respond in Turkish."
        elif language == "en":
            language_instruction = " Respond in English."
        else:
            language_instruction = f" Respond in the language with code '{language}'."
        
        if policy_type == "user_policy":
            context = (
                "You are providing feedback to a user whose input violated a safety policy. "
                "Be helpful and constructive. Explain what was wrong and suggest how they can "
                "rephrase or modify their request to comply with the policy."
            )
        else:  # agent_policy
            context = (
                "You are providing feedback to an AI agent whose response violated a safety policy. "
                "The agent will receive this feedback and try again. Be clear about what was wrong "
                "and provide specific guidance on how to generate a compliant response."
            )
        
        # Truncate content if too long
        content_preview = original_content[:500] + "..." if len(original_content) > 500 else original_content
        
        prompt = (
            f"{context}{language_instruction}\n\n"
            f"Policy Name: {policy_name}\n"
            f"Violation Reason: {violation_reason}\n"
            f"Action That Would Be Taken: {action_type}\n"
            f"Content Preview: {content_preview}\n\n"
            f"Generate constructive feedback that:\n"
            f"1. Explains clearly what aspect of the content violated the policy\n"
            f"2. Provides specific suggestions for how to modify the content\n"
            f"3. Is professional and helpful, not harsh or accusatory\n"
            f"4. Does NOT include the original problematic content in your response"
        )
        
        task = Task(prompt, response_format=PolicyFeedbackResponse)
        
        try:
            result = await self.agent.do_async(task)
            # Combine feedback message with suggested approach for comprehensive feedback
            return f"{result.feedback_message}\n\nSuggested approach: {result.suggested_approach}"
        except Exception:
            # Fallback to generic feedback message
            return (
                f"Your content was flagged by the '{policy_name}' policy. "
                f"Reason: {violation_reason}. Please modify your content to comply with this policy."
            )