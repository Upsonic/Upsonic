import pyautogui
from .signal import *
import threading

try:

    from ..screen.shot import *
    from ..agent.process import *
    from ..agent.chat_history import clear_chat_history
    from ..utils.db import (
        screenshot_path,
        save_api_key,
        load_api_key,
        activate_just_text_model,
        deactivate_just_text_model,
        is_just_text_model_active,
        set_profile,
        get_profile,
    )
    from ..screen.shot import take_screenshot
except ImportError:

    from screen.shot import *
    from agent.process import *
    from utils.db import (
        screenshot_path,
    )
    from screen.shot import take_screenshot

recording_thread = None


class ButtonHandler:
    """Handles button click events and corresponding actions."""

    def __init__(self, main_window):
        """Initialize the ButtonHandler."""
        self.recording = False
        self.main_window = main_window
        self.process_audio_thread = None

        signal_handler.recording_started.connect(self.on_recording_started)
        signal_handler.recording_stopped.connect(self.on_recording_stopped)
        signal_handler.assistant_thinking.connect(self.on_assistant_thinking)
        signal_handler.assistant_response_ready.connect(
            self.on_assistant_response_ready
        )
        signal_handler.assistant_response_stopped.connect(
            self.on_assistant_response_stopped
        )

    def toggle_recording(
        self,
        no_screenshot=False,
        take_system_audio=False,
        dont_save_image=False,
        new_record=False,
    ):
        """Toggle audio recording."""

        try:
            from ..audio.record import stop_recording, start_recording
        except ImportError:
            from audio.record import stop_recording, start_recording

        if self.recording and not new_record:
            stop_recording()
            self.recording = False
        else:
            if not no_screenshot:
                screenshot = pyautogui.screenshot()
                screenshot.save(screenshot_path)

            self.no_screenshot = no_screenshot
            self.take_system_audio = take_system_audio
            self.dont_save_image = dont_save_image

            global recording_thread
            if (
                recording_thread is None
                or not recording_thread.is_alive()
                or new_record
            ):
                recording_thread = threading.Thread(
                    target=start_recording,
                    args=(
                        take_system_audio,
                        self,
                    ),
                )
                recording_thread.start()
            signal_handler.recording_started.emit()

    def on_recording_started(self):
        """Handle event when recording starts."""

        self.recording = True
        self.main_window.update_state("talking")

    def on_recording_stopped(self):
        """Handle event when recording stops."""

        print("ON RECORDING STOPPED")
        self.recording = False
        self.main_window.update_state("thinking")
        if (
            self.process_audio_thread is None
            or not self.process_audio_thread.is_alive()
        ):
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(
                target=process_audio,
                args=(
                    not self.no_screenshot,
                    self.take_system_audio,
                    self.dont_save_image,
                ),
            )
            self.process_audio_thread.start()

    def just_screenshot(self):
        """Take a screenshot."""

        take_screenshot()
        self.process_audio_thread = threading.Thread(target=process_screenshot)
        self.process_audio_thread.start()

    def on_assistant_response_stopped(self):
        """Handle event when assistant's response stops."""

        self.main_window.update_state("idle")

    def on_assistant_thinking(self):
        """Handle event when assistant is thinking."""

        self.main_window.update_state("thinking")

    def on_assistant_response_ready(self):
        """Handle event when assistant's response is ready."""

        self.main_window.update_state("aitalking")

    def input_text(self, text):
        """Handle input text."""

        self.main_window.update_state("thinking")
        if (
            self.process_audio_thread is None
            or not self.process_audio_thread.is_alive()
        ):
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(
                target=process_text, args=(text,)
            )
            self.process_audio_thread.start()

    def input_text_screenshot(self, text):
        """Handle input text with screenshot."""

        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)

        self.main_window.update_state("thinking")
        if (
            self.process_audio_thread is None
            or not self.process_audio_thread.is_alive()
        ):
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(
                target=process_text,
                args=(text,),
                kwargs={"screenshot_path": screenshot_path},
            )
            self.process_audio_thread.start()
