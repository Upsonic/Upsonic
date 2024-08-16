try:
    from ..llm import *
    from ..utils.db import *
    from .tts_providers.openai import tts_openai
    from .tts_providers.microsoft_local import tts_microsoft_local
except ImportError:
    from llm import *
    from utils.db import *
    from audio.tts_providers.openai import tts_openai
    from audio.tts_providers.microsoft_local import tts_microsoft_local

import os
import hashlib
import random
import threading


def is_local_tts_available():
    try:
        return True
    except:
        return False


def is_openai_tts_available():
    the_model = load_model_settings()
    if llm_settings[the_model]["provider"] == "openai":
        if load_api_key() != "CHANGE_ME":
            return True
    return False


supported_openai_speakers = ["fable"]


def random_model(exclude):
    models = supported_openai_speakers.copy()
    models.remove(exclude)
    return random.choice(models)


def generate_speech_chunk(text_chunk, index, voice, results):
    sha = hashlib.sha256(text_chunk.encode()).hexdigest()
    location = os.path.join(artifacts_dir, f"{sha}.mp3")

    if os.path.exists(location):
        results[index] = location
    else:
        the_model = load_model_settings()
        tts_setting = load_tts_model_settings()
        if tts_setting == "openai":
            tts_openai(voice, text_chunk, location)

        if tts_setting == "microsoft_local":
            if not is_local_tts_available():
                print(
                    "Please install gpt-computer-assistant[local_tts] to use local TTS"
                )
            else:
                tts_microsoft_local(text_chunk, location)

        results[index] = location


def split_text_to_sentences(text, max_chunk_size=300):
    """Splits text into sentences and ensures chunks do not exceed max_chunk_size."""
    sentences = text.split(".")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def text_to_speech(text):
    text_chunks = split_text_to_sentences(text)

    threads = []
    results = [None] * len(text_chunks)

    initial_voice = random.choice(supported_openai_speakers)

    for i, chunk in enumerate(text_chunks):
        voice = (
            initial_voice if i % 2 == 0 else random_model(initial_voice)
        )  # Alternate voices
        thread = threading.Thread(
            target=generate_speech_chunk, args=(chunk, i, voice, results)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    mp3_files = [result for result in results if result is not None]

    return mp3_files
