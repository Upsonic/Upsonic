
import struct

from ..utils.db import load_pvporcupine_api_key


def wake_word(the_main_window):
    import pvporcupine
    import pyaudio

    porcupine = pvporcupine.create(access_key=load_pvporcupine_api_key(),
                                           keywords=pvporcupine.KEYWORDS)
    # Initialize PyAudio
    pa = pyaudio.PyAudio()

    # Open an audio stream
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")

    # Continuously listen for the wake word
    while the_main_window.wake_word_active:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        # Process the audio frame and check for the wake word
        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Wake word detected!")
            return True