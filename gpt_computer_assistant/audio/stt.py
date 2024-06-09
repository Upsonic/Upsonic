try:
    from ..llm import get_client
except ImportError:
    from llm import get_client

import os
from pydub import AudioSegment


def split_audio(file_path, max_size=20 * 1024 * 1024):
    """Split an audio file into smaller parts if it exceeds a maximum size.

    Args:
        file_path (str): The path to the audio file to be split.
        max_size (int): The maximum size in bytes for each split part. Defaults to 20 MB.

    Returns:
        list: A list of tuples containing the split audio segments and their respective file paths.
    """
    audio = AudioSegment.from_wav(file_path)
    file_size = os.path.getsize(file_path)
    if file_size <= max_size:
        return [(audio, file_path)]

    # Calculate the number of parts needed
    num_parts = file_size // max_size + 1
    part_length = len(audio) // num_parts
    parts = []

    for i in range(num_parts):
        start = i * part_length
        end = (i + 1) * part_length if (i + 1) < num_parts else len(audio)
        part = audio[start:end]
        part_path = f"{file_path[:-4]}_part_{i+1}.wav"
        part.export(part_path, format="wav")
        parts.append((part, part_path))

    return parts


def speech_to_text(location):
    """Convert speech audio file to text using an external service.

    Args:
        location (str): The path to the speech audio file.

    Returns:
        str: The transcribed text from the speech audio file.
    """
    audio_parts = split_audio(location)
    transcriptions = []

    for part, part_path in audio_parts:
        with open(part_path, "rb") as audio_file:
            transcription = get_client().audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            transcriptions.append(transcription)
        os.remove(part_path)  # Clean up the temporary file immediately after processing

    # Merge transcriptions (assuming it's a list of text segments)
    full_transcription = " ".join(
        transcription.text for transcription in transcriptions
    )
    return full_transcription
