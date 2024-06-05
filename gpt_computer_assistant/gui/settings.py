from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ..utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile
from ..agent.chat_history import clear_chat_history
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject



def settings_popup(self):
        from ..gpt_computer_assistant import the_input_box
        # Create a settings dialog and inside of it create a text input about openai_api_key and a button to save it
        settings_dialog = QDialog()
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.setWindowModality(Qt.ApplicationModal)

        settings_dialog.setLayout(QVBoxLayout())



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