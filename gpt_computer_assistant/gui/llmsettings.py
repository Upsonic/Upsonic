from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ..utils.db import *
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

    groq_api_key_label = QLabel("Groq API Key")
    settings_dialog.layout().addWidget(groq_api_key_label)
    groq_api_key_input = QLineEdit()
    groq_api_key = load_groq_api_key()
    groq_api_key_input.setText(groq_api_key)
    settings_dialog.layout().addWidget(groq_api_key_input)
    groq_save_button = QPushButton("Save")

    def groq_save_api_key_(api_key):
        save_groq_api_key(api_key)
        the_input_box.setText("Saved Groq API Key")
        settings_dialog.close()

    groq_save_button.clicked.connect(
        lambda: groq_save_api_key_(groq_api_key_input.text())
    )
    settings_dialog.layout().addWidget(groq_save_button)

    def hide_openai():
        api_key_label.hide()
        api_key_input.hide()
        save_button.hide()

    def hide_groq():
        groq_api_key_label.hide()
        groq_api_key_input.hide()
        groq_save_button.hide()

    def show_openai():
        api_key_label.show()
        api_key_input.show()
        save_button.show()

    def show_groq():
        groq_api_key_label.show()
        groq_api_key_input.show()
        groq_save_button.show()

    hide_openai()
    hide_groq()

    # Add a select box with the options OpenAI and Olamo
    model_label = QLabel("Model")
    model_select = QComboBox()
    model_select.addItems(
        [
            "gpt-4o (OpenAI)",
            "Llava (Ollama)",
            "BakLLaVA (Ollama)",
            "Mixtral 8x7b (Groq)",
        ]
    )

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
    elif current_model == "mixtral-8x7b-groq":
        model_select.setCurrentIndex(3)

    if model_select.currentText() == "gpt-4o (OpenAI)":
        show_openai()

    if model_select.currentText() == "Mixtral 8x7b (Groq)":
        show_groq()

    if (
        model_select.currentText() == "Llava (Ollama)"
        or model_select.currentText() == "BakLLaVA (Ollama)"
    ):
        from ..gpt_computer_assistant import the_main_window

        the_main_window.remove_painting()

    if model_select.currentText() == "Mixtral 8x7b (Groq)":
        from ..gpt_computer_assistant import the_main_window

        the_main_window.remove_screenshot_button()

    def on_model_change():
        hide_openai()
        hide_groq()
        if model_select.currentText() == "Llava (Ollama)":
            save_model_settings("llava")
            from ..gpt_computer_assistant import the_main_window

            the_main_window.remove_painting()
            the_main_window.add_screenshot_button()

        elif model_select.currentText() == "BakLLaVA (Ollama)":
            save_model_settings("bakllava")
            from ..gpt_computer_assistant import the_main_window

            the_main_window.remove_painting()
            the_main_window.add_screenshot_button()
        elif model_select.currentText() == "gpt-4o (OpenAI)":
            show_openai()
            save_model_settings("gpt-4o")
            from ..gpt_computer_assistant import the_main_window

            the_main_window.activate_painting()
            the_main_window.add_screenshot_button()
        elif model_select.currentText() == "Mixtral 8x7b (Groq)":
            show_groq()
            save_model_settings("mixtral-8x7b-groq")
            from ..gpt_computer_assistant import the_main_window

            the_main_window.remove_painting()
            the_main_window.remove_screenshot_button()

    model_select.currentIndexChanged.connect(on_model_change)

    settings_dialog.exec_()
