from ..llm import *
from .assistant import *

from .chat_history import *

from ..audio.tts import text_to_speech

from ..audio.record import audio_data


from ..gui.signal import signal_handler

import threading


from pygame import mixer


import time
import random



from ..utils.db import system_sound_location, mic_record_location, just_screenshot_path, screenshot_path


def process_audio(take_screenshot=True, take_system_audio=False):
    global audio_data, llm_history

    

    audio_file = open(mic_record_location, "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )

    if take_system_audio:
        audio_file2 = open(system_sound_location, "rb")
        transcription2 = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file2
        )

    
    llm_input = "USER: "+transcription.text

    if take_system_audio:
        llm_input += " \n Other of USER: "+transcription2.text

    llm_output = assistant(llm_input, llm_history, client, screenshot_path=screenshot_path if take_screenshot else None)




    llm_history = llm_output


    chat_message_history.add_message(llm_output[-1])
    llm_output = llm_output[-1].content


    response_path = text_to_speech(llm_output)


    signal_handler.assistant_response_ready.emit()

    def play_audio():
        mixer.init()
        mixer.music.load(response_path)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
        signal_handler.assistant_response_stopped.emit()
    
    playback_thread = threading.Thread(target=play_audio)
    playback_thread.start()



def process_screenshot():
    global llm_history

    
    llm_input = "USER: "+"I just take a screenshot. for you to remember. Just say ok."
    print("LLM INPUT (just screenshot)", llm_input)

    llm_output = assistant(llm_input, llm_history, client, screenshot_path=just_screenshot_path)



    llm_history = llm_output


    chat_message_history.add_message(llm_output[-1])
    llm_output = llm_output[-1].content



    response_path = text_to_speech(llm_output)


    signal_handler.assistant_response_ready.emit()

    def play_audio():
        mixer.init()
        mixer.music.load(response_path)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
        signal_handler.assistant_response_stopped.emit()
    
    playback_thread = threading.Thread(target=play_audio)
    playback_thread.start()
