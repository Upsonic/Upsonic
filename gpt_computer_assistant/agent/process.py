try:
    from ..llm import *
    from .assistant import *
    from .chat_history import *
    from ..audio.tts import text_to_speech
    from ..audio.stt import speech_to_text

    from ..gui.signal import signal_handler
    from ..utils.db import *
    from ..utils.telemetry import my_tracer, os_name
except ImportError:
    from llm import *
    from agent.assistant import *
    from agent.chat_history import *
    from audio.tts import text_to_speech
    from audio.stt import speech_to_text
    from gui.signal import signal_handler
    from utils.db import *
    from utils.telemetry import my_tracer, os_name


import threading
import traceback


from pygame import mixer


import time

last_ai_response = None
user_id = load_user_id()
os_name_ = os_name()


def tts_if_you_can(
    text: str, not_threaded=False, status_edit=False, bypass_other_settings=False
):
    try:
        try:
            from ..gpt_computer_assistant import the_main_window
        except ImportError:
            from gpt_computer_assistant import the_main_window

        first_control = None
        try:
            first_control = (
                not is_just_text_model_active() and not the_main_window.api_enabled
            )
        except:
            first_control = False

        if first_control or bypass_other_settings:
            response_path = text_to_speech(text)
            if status_edit:
                signal_handler.assistant_response_ready.emit()

            def play_audio():
                for each_r in response_path:
                    mixer.init()
                    mixer.music.load(each_r)
                    mixer.music.play()
                    while mixer.music.get_busy():
                        the_stop_talking = False
                        try:
                            the_stop_talking = the_main_window.stop_talking
                        except:
                            pass
                        if the_stop_talking:
                            mixer.music.stop()
                            break
                        time.sleep(0.1)
                if status_edit:
                    signal_handler.assistant_response_stopped.emit()

            if not not_threaded:
                playback_thread = threading.Thread(target=play_audio)
                playback_thread.start()
            else:
                play_audio()
    except Exception:
        traceback.print_exc()
        pass


def process_audio(take_screenshot=True, take_system_audio=False, dont_save_image=False):
    with my_tracer.start_span("process_audio") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)
        try:
            from ..audio.record import audio_data
            global audio_data, last_ai_response
            from ..gpt_computer_assistant import the_input_box, the_main_window
            from ..audio.record import audio_data
            from ..audio.input_box import the_input_box_pre

            transcription = speech_to_text(mic_record_location)

            if take_system_audio:
                transcription2 = speech_to_text(system_sound_location)

            llm_input = transcription

            print("Previously AI response", last_ai_response, "end prev")

            print("Input Box AI", the_input_box_pre)

            if (
                the_input_box_pre != ""
                and not the_input_box_pre.startswith("System:")
                and the_input_box_pre not in last_ai_response
            ):
                llm_input += the_input_box_pre

            if take_system_audio:
                llm_input += " \n Other of USER: " + transcription2

            if the_input_box.toPlainText().startswith("System:"):
                the_main_window.update_from_thread(
                    "Transciption Completed. Running AI..."
                )

            print("LLM INPUT (screenshot)", llm_input)

            llm_output = assistant(
                llm_input,
                get_client(),
                screenshot_path=screenshot_path if take_screenshot else None,
                dont_save_image=dont_save_image,
            )
            if the_input_box.toPlainText().startswith("System:"):
                the_main_window.update_from_thread(
                    "AI Response Completed. Generating Audio..."
                )
            last_ai_response = llm_output.replace("<Answer>", "")

            from ..gpt_computer_assistant import the_main_window

            model = load_model_settings()
            if not llm_settings[model][
                "stream"
            ] or the_main_window.worker.the_input_text.startswith("System:"):
                the_main_window.set_text_to_input_box(last_ai_response)
                the_main_window.complated_answer = True

            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_main_window

                the_main_window.complated_answer = True
                the_main_window.manuel_stop = True
                while (
                    the_main_window.reading_thread or the_main_window.reading_thread_2
                ):
                    time.sleep(0.1)
                the_main_window.read_part_task()
                if the_main_window.stop_talking:
                    the_main_window.stop_talking = False
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()
        except Exception as e:
            print("Error in process_audio", e)
            traceback.print_exc()
            from ..gpt_computer_assistant import the_input_box, the_main_window

            exception_str = traceback.format_exc()

            the_main_window.update_from_thread("EXCEPTION: " + str(exception_str))
            tts_if_you_can("Exception occurred. Please check the logs.")
            signal_handler.assistant_response_stopped.emit()


def process_screenshot():
    with my_tracer.start_span("process_screenshot") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)
        try:
            global last_ai_response
            from ..gpt_computer_assistant import the_input_box, the_main_window

            from ..audio.input_box import the_input_box_pre

            llm_input = "I just take a screenshot. for you to remember. Just say 'Ok.' if the user doesnt want anything before."

            if (
                the_input_box_pre != ""
                and not the_input_box_pre.startswith("System:")
                and the_input_box_pre not in last_ai_response
            ):
                llm_input += the_input_box_pre

            print("LLM INPUT (just screenshot)", llm_input)

            if the_input_box.toPlainText().startswith("System:"):
                the_main_window.update_from_thread(
                    "Transciption Completed. Running AI..."
                )

            llm_output = assistant(
                llm_input,
                get_client(),
                screenshot_path=just_screenshot_path,
                dont_save_image=False,
                just_screenshot=True,
            )

            last_ai_response = llm_output.replace("<Answer>", "")

            from ..gpt_computer_assistant import the_main_window

            model = load_model_settings()
            if not llm_settings[model][
                "stream"
            ] or the_main_window.worker.the_input_text.startswith("System:"):
                the_main_window.set_text_to_input_box(last_ai_response)
                the_main_window.complated_answer = True

            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_main_window

                the_main_window.complated_answer = True
                the_main_window.manuel_stop = True
                while (
                    the_main_window.reading_thread or the_main_window.reading_thread_2
                ):
                    time.sleep(0.1)
                the_main_window.read_part_task()
                if the_main_window.stop_talking:
                    the_main_window.stop_talking = False
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()

        except Exception as e:
            print("Error in process_screenshot", e)
            traceback.print_exc()
            from ..gpt_computer_assistant import the_input_box, the_main_window

            exception_str = traceback.format_exc()

            the_main_window.update_from_thread("EXCEPTION: " + str(exception_str))
            tts_if_you_can("Exception occurred. Please check the logs.")
            signal_handler.assistant_response_stopped.emit()


def process_text(text, screenshot_path=None):
    with my_tracer.start_span("process_text") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)
        try:
            global last_ai_response

            llm_input = text

            llm_output = assistant(
                llm_input,
                get_client(),
                screenshot_path=screenshot_path,
                dont_save_image=True,
            )
            last_ai_response = llm_output.replace("<Answer>", "")

            from ..gpt_computer_assistant import the_main_window

            model = load_model_settings()
            if not llm_settings[model][
                "stream"
            ] or the_main_window.worker.the_input_text.startswith("System:"):
                the_main_window.set_text_to_input_box(last_ai_response)
                the_main_window.complated_answer = True

            signal_handler.assistant_response_ready.emit()

            def play_text():
                from ..gpt_computer_assistant import the_main_window

                the_main_window.complated_answer = True
                the_main_window.manuel_stop = True
                while (
                    the_main_window.reading_thread or the_main_window.reading_thread_2
                ):
                    time.sleep(0.1)
                the_main_window.read_part_task()
                if the_main_window.stop_talking:
                    the_main_window.stop_talking = False
                signal_handler.assistant_response_stopped.emit()

            playback_thread = threading.Thread(target=play_text)
            playback_thread.start()

        except Exception as e:
            print("Error in process_text", e)
            traceback.print_exc()
            from ..gpt_computer_assistant import the_main_window

            exception_str = traceback.format_exc()

            the_main_window.update_from_thread("EXCEPTION: " + str(exception_str))
            tts_if_you_can("Exception occurred. Please check the logs.")
            signal_handler.assistant_response_stopped.emit()



def process_text_api(text, screenshot_path=None):
    with my_tracer.start_span("process_text_api") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)
        try:
            global last_ai_response

            llm_input = text

            llm_output = assistant(
                llm_input,
                get_client(),
                screenshot_path=screenshot_path,
                dont_save_image=True,
            )

            return llm_output


        except Exception as e:
            print("Error in process_text", e)
            traceback.print_exc()
            
