import pyautogui
from .signal import *
from ..audio.record import *
from ..screen.shot import *
from ..agent.proccess import *
from ..agent.chat_history import clear_chat_history
from ..utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile
from ..screen.shot import take_screenshot
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

recording_thread = None


class ButtonHandler():
    def __init__(self, main_window):
        self.recording = False
        self.main_window = main_window
        self.process_audio_thread = None

        signal_handler.recording_started.connect(self.on_recording_started)
        signal_handler.recording_stopped.connect(self.on_recording_stopped)
        signal_handler.assistant_thinking.connect(self.on_assistant_thinking)
        signal_handler.assistant_response_ready.connect(self.on_assistant_response_ready)
        signal_handler.assistant_response_stopped.connect(self.on_assistant_response_stopped)


    def toggle_recording(self, no_screenshot=False, take_system_audio=False, dont_save_image=False):
        
        if self.recording:
            stop_recording()
            self.recording = False
        else:
            # Take screenshot before starting recording
            if not no_screenshot:
                screenshot = pyautogui.screenshot()

                screenshot.save(screenshot_path)

            
            self.no_screenshot = no_screenshot

            self.take_system_audio = take_system_audio
            self.dont_save_image = dont_save_image

            global recording_thread
            if recording_thread is None or not recording_thread.is_alive():
                recording_thread = threading.Thread(target=start_recording, args=(take_system_audio,))
                recording_thread.start()
            signal_handler.recording_started.emit()

    def on_recording_started(self):
        self.recording = True
        self.main_window.update_state('talking')
    def on_recording_stopped(self):
        print("ON RECORDING STOPPED")
        self.recording = False
        self.main_window.update_state('thinking')
        if self.process_audio_thread is None or not self.process_audio_thread.is_alive():
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(target=process_audio, args=(not self.no_screenshot,self.take_system_audio, self.dont_save_image))
            self.process_audio_thread.start()



    def just_screenshot(self):
        
        take_screenshot()
        self.process_audio_thread = threading.Thread(target=process_screenshot)
        self.process_audio_thread.start()







    def on_assistant_response_stopped(self):
        self.main_window.update_state('idle')

    def on_assistant_thinking(self):
        self.main_window.update_state('thinking')

    def on_assistant_response_ready(self):
        self.main_window.update_state('talking')



    def input_text(self, text):
        
        self.main_window.update_state('thinking')
        if self.process_audio_thread is None or not self.process_audio_thread.is_alive():
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(target=process_text, args=(text,))
            self.process_audio_thread.start()


    def input_text_screenshot(self, text):
        screenshot = pyautogui.screenshot()

        screenshot.save(screenshot_path)

            

        self.main_window.update_state('thinking')
        if self.process_audio_thread is None or not self.process_audio_thread.is_alive():
            signal_handler.assistant_thinking.emit()
            self.process_audio_thread = threading.Thread(target=process_text, args=(text,), kwargs={"screenshot_path":screenshot_path})
            self.process_audio_thread.start()            



























