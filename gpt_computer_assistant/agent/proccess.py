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
from ..utils.db import *





from ..utils.db import system_sound_location, mic_record_location, just_screenshot_path, screenshot_path


last_ai_response = None



def process_audio(take_screenshot=True, take_system_audio=False, dont_save_image=False):
    global audio_data, last_ai_response

    

    transcription = speech_to_text(mic_record_location)

    if take_system_audio:

        transcription2 = speech_to_text(system_sound_location)

    
    llm_input = "USER: "+transcription

    if take_system_audio:
        llm_input += " \n Other of USER: "+transcription2

    llm_output = assistant(llm_input, get_chat_message_history().messages, get_client(), screenshot_path=screenshot_path if take_screenshot else None)



    if dont_save_image:
        currently_messages = get_chat_message_history().messages
        if take_screenshot:
            last_message = currently_messages[-1].content[0]
            currently_messages.remove(currently_messages[-1])

            get_chat_message_history().clear()
            for message in currently_messages:
                get_chat_message_history().add_message(message)
            get_chat_message_history().add_message(HumanMessage(content=[last_message]))

    get_chat_message_history().add_message(llm_output[-1])
    llm_output = llm_output[-1].content

    print("Whole LLM OUTPUT", get_chat_message_history().messages)
    
    signal_handler.assistant_response_ready.emit()


    

    if not is_just_text_model_active():
        response_path = text_to_speech(llm_output)

        def play_text():
            from ..gpt_computer_assistant import the_input_box
            global last_ai_response
            if the_input_box.text() == "" or the_input_box.text() == last_ai_response:
                the_input_box.setText(llm_output)
                last_ai_response = llm_output
         

        def play_audio():
            play_text()
            mixer.init()
            mixer.music.load(response_path)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            signal_handler.assistant_response_stopped.emit()
        


        playback_thread = threading.Thread(target=play_audio)
        playback_thread.start()
    else:
        def play_text():
            from ..gpt_computer_assistant import the_input_box
            the_input_box.setText(llm_output)
            signal_handler.assistant_response_stopped.emit()

        playback_thread = threading.Thread(target=play_text)
        playback_thread.start()



def process_screenshot():
    global last_ai_response

    
    llm_input = "USER: "+"I just take a screenshot. for you to remember. Just say ok."
    print("LLM INPUT (just screenshot)", llm_input)

    llm_output = assistant(llm_input, get_chat_message_history().messages, get_client(), screenshot_path=just_screenshot_path)






    get_chat_message_history().add_message(llm_output[-1])
    llm_output = llm_output[-1].content


   


    signal_handler.assistant_response_ready.emit()

    if not is_just_text_model_active():
        response_path = text_to_speech(llm_output)
        def play_text():
            from ..gpt_computer_assistant import the_input_box
            global last_ai_response
            if the_input_box.text() == "" or the_input_box.text() == last_ai_response:
                the_input_box.setText(llm_output)
                last_ai_response = llm_output
         

        def play_audio():
            play_text()
            mixer.init()
            mixer.music.load(response_path)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            signal_handler.assistant_response_stopped.emit()
        

        
        playback_thread = threading.Thread(target=play_audio)
        playback_thread.start()
    else:
        def play_text():
            from ..gpt_computer_assistant import the_input_box
            the_input_box.setText(llm_output)
            signal_handler.assistant_response_stopped.emit()

        playback_thread = threading.Thread(target=play_text)
        playback_thread.start()
        



def process_text(text, screenshot_path=None):
    global last_ai_response

    
    llm_input = "USER: "+text



    llm_output = assistant(llm_input, get_chat_message_history().messages, get_client(), screenshot_path=screenshot_path)


    # Remove the image
    currently_messages = get_chat_message_history().messages
    last_message = currently_messages[-1].content[0]
    currently_messages.remove(currently_messages[-1])

    get_chat_message_history().clear()
    for message in currently_messages:
        get_chat_message_history().add_message(message)
    get_chat_message_history().add_message(HumanMessage(content=[last_message]))






    get_chat_message_history().add_message(llm_output[-1])
    llm_output = llm_output[-1].content

    

    signal_handler.assistant_response_ready.emit()

    if not is_just_text_model_active():
        response_path = text_to_speech(llm_output)
        def play_text():
            from ..gpt_computer_assistant import the_input_box
            global last_ai_response
            if the_input_box.text() == "" or the_input_box.text() == last_ai_response:
                the_input_box.setText(llm_output)
                last_ai_response = llm_output
         

        def play_audio():
            play_text()
            mixer.init()
            mixer.music.load(response_path)
            mixer.music.play()
            while mixer.music.get_busy():
                time.sleep(0.1)
            signal_handler.assistant_response_stopped.emit()
        

        
        playback_thread = threading.Thread(target=play_audio)
        playback_thread.start()
    else:
        def play_text():
            from ..gpt_computer_assistant import the_input_box
            the_input_box.setText(llm_output)
            signal_handler.assistant_response_stopped.emit()

        playback_thread = threading.Thread(target=play_text)
        playback_thread.start()
        



