try:
    from ...llm import *
except ImportError:
    from llm import *


def stt_openai(audio_file):
    transcription = get_client().audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return transcription.text
