from ..llm import *
from .assistant import *

from .chat_history import *

from ..audio.tts import text_to_speech
from ..audio.stt import speech_to_text

from ..audio.record import audio_data


from ..gui.signal import signal_handler

import threading


from pygame import mixer


import time
import random



from ..utils.db import system_sound_location, mic_record_location, just_screenshot_path, screenshot_path


def process_audio(take_screenshot=True, take_system_audio=False):
    global audio_data

    

    transcription = speech_to_text(mic_record_location)

    if take_system_audio:

        transcription2 = speech_to_text(system_sound_location)

    
    llm_input = "USER: "+transcription.text

    if take_system_audio:
        llm_input += " \n Other of USER: "+transcription2.text

    llm_output = assistant(llm_input, chat_message_history.messages, get_client(), screenshot_path=screenshot_path if take_screenshot else None)





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


    
    llm_input = "USER: "+"I just take a screenshot. for you to remember. Just say ok."
    print("LLM INPUT (just screenshot)", llm_input)

    llm_output = assistant(llm_input, chat_message_history.messages, get_client(), screenshot_path=just_screenshot_path)






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




def process_text(text):


    
    llm_input = "USER: "+text



    llm_output = assistant(llm_input, chat_message_history.messages, get_client(), screenshot_path=None)





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