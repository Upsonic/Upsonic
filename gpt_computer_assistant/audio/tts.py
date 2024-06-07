try:
    from ..llm import *
    from ..utils.db import artifacts_dir
except ImportError:
    from llm import *
    from utils.db import artifacts_dir
import os

import hashlib


def text_to_speech(text):
    # create sha256 hash of text and save it if already exists just return it
    # if not exists create it and save it

    sha = hashlib.sha256(text.encode()).hexdigest()

    location = os.path.join(artifacts_dir, f"{sha}.mp3")

    if os.path.exists(location):
        return location
    else:
        response = get_client().audio.speech.create(
            model="tts-1",
            voice="fable",
            input=text,
        )

        response.stream_to_file(location)
        return location
