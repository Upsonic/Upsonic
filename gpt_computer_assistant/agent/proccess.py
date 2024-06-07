try:
    from ..llm import *
    from .assistant import *
    from .chat_history import *
    from ..audio.tts import text_to_speech
    from ..audio.stt import speech_to_text
    from ..audio.record import audio_data
    from ..gui.signal import signal_handler
    from ..utils.db import *
    from ..utils.telemetry import my_tracer, os_name
except ImportError:
    from llm import *
    from agent.assistant import *
    from agent.chat_history import *
    from audio.tts import text_to_speech
    from audio.stt import speech_to_text
    from audio.record import audio_data
    from gui.signal import signal_handler
    from utils.db import *
    from utils.telemetry import my_tracer, os_name


import threading


from pygame import mixer


import time
import random

last_ai_response = None
user_id = load_user_id()
os_name_ = os_name()


def process_audio(take_screenshot=True, take_system_audio=False, dont_save_image=False):
    with my_tracer.start_span("process_audio") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)

        global audio_data, last_ai_response

        transcription = speech_to_text(mic_record_location)

        if take_system_audio:

            transcription2 = speech_to_text(system_sound_location)

        llm_input = "USER: " + transcription

        if take_system_audio:
            llm_input += " \n Other of USER: " + transcription2

        llm_output = assistant(
            llm_input,
            get_chat_message_history().messages,
            get_client(),
            screenshot_path=screenshot_path if take_screenshot else None,
            dont_save_image=dont_save_image,
        )

        if not is_just_text_model_active():
            response_path = text_to_speech(llm_output)
            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_input_box

                global last_ai_response
                if (
                    the_input_box.text() == ""
                    or the_input_box.text() == "Thinking..."
                    or the_input_box.text() == last_ai_response
                ):
                    the_input_box.setText(llm_output)
                    last_ai_response = llm_output

            def play_audio():
                with my_tracer.start_span("play_audio") as span:
                    span.set_attribute("user_id", user_id)
                    span.set_attribute("os_name", os_name_)
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
            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_input_box
                the_input_box.setText(llm_output)
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()


def process_screenshot():
    with my_tracer.start_span("process_screenshot") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)

        global last_ai_response

        llm_input = (
            "USER: " + "I just take a screenshot. for you to remember. Just say ok."
        )
        print("LLM INPUT (just screenshot)", llm_input)

        llm_output = assistant(
            llm_input,
            get_chat_message_history().messages,
            get_client(),
            screenshot_path=just_screenshot_path,
            dont_save_image=True,
        )

        if not is_just_text_model_active():
            response_path = text_to_speech(llm_output)
            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_input_box

                global last_ai_response
                if (
                    the_input_box.text() == ""
                    or the_input_box.text() == "Thinking..."
                    or the_input_box.text() == last_ai_response
                ):
                    the_input_box.setText(llm_output)
                    last_ai_response = llm_output

            def play_audio():
                with my_tracer.start_span("play_audio") as span:
                    span.set_attribute("user_id", user_id)
                    span.set_attribute("os_name", os_name_)
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
            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_input_box

                the_input_box.setText(llm_output)
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()


def process_text(text, screenshot_path=None):
    with my_tracer.start_span("process_text") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)

        global last_ai_response

        llm_input = "USER: " + text

        llm_output = assistant(
            llm_input,
            get_chat_message_history().messages,
            get_client(),
            screenshot_path=screenshot_path,
            dont_save_image=True,
        )

        if not is_just_text_model_active():

            def play_text():
                from ..gpt_computer_assistant import the_input_box

                global last_ai_response
                if (
                    the_input_box.text() == ""
                    or the_input_box.text() == "Thinking..."
                    or the_input_box.text() == last_ai_response
                ):
                    the_input_box.setText(llm_output)
                    last_ai_response = llm_output

            if load_api_key() != "CHANGE_ME":
                response_path = text_to_speech(llm_output)
                signal_handler.assistant_response_ready.emit()

                def play_audio():
                    with my_tracer.start_span("play_audio") as span:
                        span.set_attribute("user_id", user_id)
                        span.set_attribute("os_name", os_name_)
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
                signal_handler.assistant_response_ready.emit()
                play_text()
                signal_handler.assistant_response_stopped.emit()

        else:
            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_input_box

                the_input_box.setText(llm_output)
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()
