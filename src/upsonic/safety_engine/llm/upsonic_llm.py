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