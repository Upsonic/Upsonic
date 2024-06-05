from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ..utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile, load_model_settings, save_model_settings
from ..agent.chat_history import clear_chat_history
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt5.QtWidgets import QComboBox, QLabel


def llmsettings_popup(self):
        from ..gpt_computer_assistant import the_input_box
        # Create a settings dialog and inside of it create a text input about openai_api_key and a button to save it
        settings_dialog = QDialog()
        settings_dialog.setWindowTitle("Settings")
        settings_dialog.setWindowModality(Qt.ApplicationModal)

        settings_dialog.setLayout(QVBoxLayout())
        api_key_label = QLabel("OpenAI API Key")
        settings_dialog.layout().addWidget(api_key_label)
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




        # Add a select box with the options OpenAI and Olamo
        model_label = QLabel("Model")
        model_select = QComboBox()
        model_select.addItems(["gpt-4o (OpenAI)", "Llava (Ollama)", "BakLLaVA (Ollama)"])

        settings_dialog.layout().addWidget(model_label)
        settings_dialog.layout().addWidget(model_select)   

        
        # currently model
        current_model = load_model_settings()
        if current_model == "gpt-4o":
            model_select.setCurrentIndex(0)
        elif current_model == "llava":
            model_select.setCurrentIndex(1)
        elif current_model == "bakllava":
            model_select.setCurrentIndex(2)

        if model_select.currentText() == "Llava (Ollama)":
            api_key_label.hide()
            api_key_input.hide()
            save_button.hide()
        elif model_select.currentText() == "BakLLaVA (Ollama)":
            api_key_label.hide()
            api_key_input.hide()
            save_button.hide()


        def on_model_change():
            if model_select.currentText() == "Llava (Ollama)":
                api_key_label.hide()
                api_key_input.hide()
                save_button.hide()
                save_model_settings("llava")
                from ..gpt_computer_assistant import the_main_window
                the_main_window.remove_painting()

            elif model_select.currentText() == "BakLLaVA (Ollama)":
                api_key_label.hide()
                api_key_input.hide()
                save_button.hide()
                save_model_settings("bakllava")
                from ..gpt_computer_assistant import the_main_window
                the_main_window.remove_painting()
            else:
                api_key_label.show()
                api_key_input.show()
                save_button.show()
                save_model_settings("gpt-4o")
                from ..gpt_computer_assistant import the_main_window
                the_main_window.activate_painting()                

        model_select.currentIndexChanged.connect(on_model_change)



        
        settings_dialog.exec_()