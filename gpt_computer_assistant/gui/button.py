
from .signal import *

from ..audio.record import *

from ..screen.shot import *

from ..agent.proccess import *

from ..agent.chat_history import clear_chat_history

import pyautogui
recording_thread = None


from ..utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile
from ..screen.shot import take_screenshot




from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton


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




    def settings_popup(self):
        from ..gpt_computer_assistant import the_input_box
        # Create a settings dialog and inside of it create a text input about openai_api_key and a button to save it
        settings_dialog = QDialog()
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.setWindowModality(Qt.ApplicationModal)

        settings_dialog.setLayout(QVBoxLayout())
        settings_dialog.layout().addWidget(QLabel("OpenAI API Key Get It Here https://platform.openai.com/api-keys"))
        api_key_input = QLineEdit()
        api_key = load_api_key()
        api_key_input.setText(api_key)
        settings_dialog.layout().addWidget(api_key_input)
        save_button = QPushButton("Save")

        def save_api_key_(api_key):
            save_api_key(api_key)
            the_input_box.setText("Saved API Key")
            settings_dialog.close()

        save_button.clicked.connect(lambda: save_api_key_(api_key_input.text()))
        settings_dialog.layout().addWidget(save_button)

        # Add another button to reset memory
        reset_memory_button = QPushButton("Reset Memory")

        def clear_chat_history_():
            clear_chat_history()
            the_input_box.setText("Cleared Chat History")
            settings_dialog.close()

        reset_memory_button.clicked.connect(clear_chat_history_)
        settings_dialog.layout().addWidget(reset_memory_button)

        # Add another button to enable just text model
        just_text_button = QPushButton("Enable Just Text Model")

        settings_dialog.layout().addWidget(just_text_button)

        if is_just_text_model_active():
            just_text_button.setText("Disabled Just Text Model")

            def deactivate_just_text_model_():
                deactivate_just_text_model()
                the_input_box.setText("Disabled Just Text Model")
                settings_dialog.close()

            just_text_button.clicked.connect(deactivate_just_text_model_)
        else:

            def activate_just_text_model_():
                activate_just_text_model()
                the_input_box.setText("Enabled Just Text Model")
                settings_dialog.close()

            just_text_button.clicked.connect(activate_just_text_model_)


        #create a input form and save button to change profile
        settings_dialog.layout().addWidget(QLabel("Profile"))
        profile_input = QLineEdit()

        profile_input.setText(get_profile())
        settings_dialog.layout().addWidget(profile_input)
        profile_save_button = QPushButton("Save")

        def set_profile_(profile):
            set_profile(profile)
            the_input_box.setText("Saved Profile")
            settings_dialog.close()

        profile_save_button.clicked.connect(lambda: set_profile_(profile_input.text()))
        settings_dialog.layout().addWidget(profile_save_button)



        
        settings_dialog.exec_()
    



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



























