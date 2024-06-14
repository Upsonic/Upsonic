try:
    from ..llm import *
    from ..utils.db import artifacts_dir
except ImportError:
    from llm import *
    from utils.db import artifacts_dir

import os
import hashlib
import random
import threading

supported_openai_speakers = ["fable", "nova"]

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
        response = get_client().audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text_chunk,
        )
        response.stream_to_file(location)
        results[index] = location

def split_text_to_sentences(text, max_chunk_size=300):
    """Splits text into sentences and ensures chunks do not exceed max_chunk_size."""
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (sentence + '. ')
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def text_to_speech(text):
    text_chunks = split_text_to_sentences(text)
    
    threads = []
    results = [None] * len(text_chunks)
    
    initial_voice = random.choice(supported_openai_speakers)
    
    for i, chunk in enumerate(text_chunks):
        voice = initial_voice if i % 2 == 0 else random_model(initial_voice)  # Alternate voices
        thread = threading.Thread(
            target=generate_speech_chunk,
            args=(chunk, i, voice, results)
        )
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    mp3_files = [result for result in results if result is not None]

    return mp3_files

