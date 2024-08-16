try:
    from ...llm import *
except ImportError:
    from llm import *


def tts_openai(voice, text_chunk, location):
    response = get_client().audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text_chunk,
    )
    response.stream_to_file(location)
