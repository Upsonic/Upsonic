try:
    from ..utils.db import *
    from ..agent.chat_history import clear_chat_history
    from ..llm_settings import llm_show_name, llm_settings
    from ..audio.tts import is_local_tts_available
    from ..audio.stt import is_local_stt_available

except ImportError:
    from utils.db import *
    from llm_settings import llm_show_name, llm_settings
    from audio.tts import is_local_tts_available
    from audio.stt import is_local_stt_available

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
)
from PyQt5.QtCore import Qt


def llmsettings_popup(self):
    from ..gpt_computer_assistant import the_main_window

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
        the_main_window.update_from_thread("Saved API Key")
        the_main_window.input_box.setPlaceholderText("Type here")
        settings_dialog.close()

    save_button.clicked.connect(lambda: save_api_key_(api_key_input.text()))
    settings_dialog.layout().addWidget(save_button)

    openai_url_label = QLabel("OpenAI Base URL")
    settings_dialog.layout().addWidget(openai_url_label)
    openai_url_input = QLineEdit()
    openai_url = load_openai_url()
    openai_url_input.setText(openai_url)
    settings_dialog.layout().addWidget(openai_url_input)

    def save_openai_url_():
        openai_url = openai_url_input.text()
        save_openai_url(openai_url)
        the_main_window.update_from_thread("Saved OpenAI Base URL")
        the_main_window.input_box.setPlaceholderText("Type here")
        settings_dialog.close()

    openai_url_save_button = QPushButton("Save URL")
    openai_url_save_button.clicked.connect(save_openai_url_)
    settings_dialog.layout().addWidget(openai_url_save_button)




    api_version_label = QLabel("API Version")
    settings_dialog.layout().addWidget(api_version_label)
    api_version_input = QLineEdit()
    api_version = load_api_version()
    api_version_input.setText(api_version)
    settings_dialog.layout().addWidget(api_version_input)

    def save_api_version_():
        api_version = api_version_input.text()
        save_api_version(api_version)
        the_main_window.update_from_thread("Saved API Version")
        the_main_window.input_box.setPlaceholderText("Type here")
        settings_dialog.close()

    api_version_save_button = QPushButton("Save URL")
    api_version_save_button.clicked.connect(save_api_version_)
    settings_dialog.layout().addWidget(api_version_save_button)


    

    groq_api_key_label = QLabel("Groq API Key")
    settings_dialog.layout().addWidget(groq_api_key_label)
    groq_api_key_input = QLineEdit()
    groq_api_key = load_groq_api_key()
    groq_api_key_input.setText(groq_api_key)
    settings_dialog.layout().addWidget(groq_api_key_input)
    groq_save_button = QPushButton("Save")

    def groq_save_api_key_(api_key):
        save_groq_api_key(api_key)
        the_main_window.update_from_thread("Saved Groq API Key")
        the_main_window.input_box.setPlaceholderText("Type here")
        settings_dialog.close()

    groq_save_button.clicked.connect(
        lambda: groq_save_api_key_(groq_api_key_input.text())
    )
    settings_dialog.layout().addWidget(groq_save_button)

    google_api_key_label = QLabel("Google Generative AI API Key")
    settings_dialog.layout().addWidget(google_api_key_label)
    google_api_key_input = QLineEdit()
    google_api_key = load_google_api_key()
    google_api_key_input.setText(google_api_key)
    settings_dialog.layout().addWidget(google_api_key_input)
    google_save_button = QPushButton("Save")

    def google_save_api_key_(api_key):
        save_google_api_key(api_key)
        the_main_window.update_from_thread("Saved Google API Key")
        the_main_window.input_box.setPlaceholderText("Type here")
        settings_dialog.close()

    google_save_button.clicked.connect(
        lambda: google_save_api_key_(google_api_key_input.text())
    )
    settings_dialog.layout().addWidget(google_save_button)

    def hide_openai():
        api_key_label.hide()
        api_key_input.hide()

        save_button.hide()


    def hide_azureai():
        api_key_label.hide()
        api_key_input.hide()
        save_button.hide()
        openai_url_label.hide()
        openai_url_input.hide()
        openai_url_save_button.hide()
        api_version_label.hide()
        api_version_input.hide()
        api_version_save_button.hide()
        

    def hide_groq():
        groq_api_key_label.hide()
        groq_api_key_input.hide()
        groq_save_button.hide()

    def hide_google():
        google_api_key_label.hide()
        google_api_key_input.hide()
        google_save_button.hide()

    def show_openai():
        api_key_label.show()
        api_key_input.show()
        save_button.show()

    def show_azureai():
        api_key_label.show()
        api_key_input.show()
        save_button.show()
        openai_url_label.show()
        openai_url_input.show()
        openai_url_save_button.show()
        api_version_label.show()
        api_version_input.show()
        api_version_save_button.show()

    def show_groq():
        groq_api_key_label.show()
        groq_api_key_input.show()
        groq_save_button.show()

    def show_google():
        google_api_key_label.show()
        google_api_key_input.show()
        google_save_button.show()

    hide_openai()
    hide_azureai()
    hide_groq()
    hide_google()

    model_label = QLabel("Model")
    model_select = QComboBox()
    model_select.addItems(list(llm_show_name.keys()))
    settings_dialog.layout().addWidget(model_label)
    settings_dialog.layout().addWidget(model_select)

    current_model = load_model_settings()
    for i, model in enumerate(llm_show_name.keys()):
        if llm_show_name[model] == current_model:
            model_select.setCurrentIndex(i)

    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "openai":
        show_openai()

    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "azureai":
        show_azureai()

    
    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "groq":
        show_groq()
    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "google":
        show_google()

    if not llm_settings[llm_show_name[model_select.currentText()]]["vision"]:
        the_main_window.remove_screenshot_button()

    def on_model_change():
        hide_openai()
        hide_azureai()
        hide_groq()
        hide_google()
        the_save_string = llm_show_name[model_select.currentText()]
        save_model_settings(the_save_string)

        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "openai"
        ):
            show_openai()

        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "anthropic"
        ):
            show_openai()

        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "azureai"
        ):
            show_azureai()

        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "azureopenai"
        ):
            show_openai()
            openai_url_label.show()
            openai_url_input.show()
            openai_url_save_button.show()

        if llm_settings[llm_show_name[model_select.currentText()]]["vision"]:
            the_main_window.add_screenshot_button()
        else:
            the_main_window.remove_screenshot_button()
        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "groq"
        ):
            show_groq()
        if (
            llm_settings[llm_show_name[model_select.currentText()]]["provider"]
            == "google"
        ):
            show_google()

    model_select.currentIndexChanged.connect(on_model_change)

    # Add TTS model selection
    tts_model_label = QLabel("TTS Model")
    tts_model_select = QComboBox()
    tts_model_select.addItems(["openai", "microsoft_local"])
    settings_dialog.layout().addWidget(tts_model_label)
    settings_dialog.layout().addWidget(tts_model_select)

    currently_tts_model = load_tts_model_settings()

    if currently_tts_model == "openai":
        tts_model_select.setCurrentIndex(0)
        show_openai()
    else:
        tts_model_select.setCurrentIndex(1)

    def on_tts_model_change():
        if tts_model_select.currentText() == "openai":
            show_openai()
            save_tts_model_settings("openai")
        else:
            save_tts_model_settings("microsoft_local")

    if not is_local_tts_available():
        # add an text to inform the user that local tts is not available
        information_text = QLabel(
            "Please install gpt-computer-assistant[local_tts] to use local TTS"
        )
        settings_dialog.layout().addWidget(information_text)
        tts_model_select.setEnabled(False)

    tts_model_select.currentIndexChanged.connect(on_tts_model_change)

    # Add STT model selection
    stt_model_label = QLabel("STT Model")
    stt_model_select = QComboBox()
    stt_model_select.addItems(["openai", "openai_whisper_local"])
    settings_dialog.layout().addWidget(stt_model_label)
    settings_dialog.layout().addWidget(stt_model_select)

    currently_stt_model = load_stt_model_settings()

    if currently_stt_model == "openai":
        stt_model_select.setCurrentIndex(0)
        show_openai()
    else:
        stt_model_select.setCurrentIndex(1)

    def on_stt_model_change():
        if stt_model_select.currentText() == "openai":
            show_openai()
            save_stt_model_settings("openai")
        else:
            save_stt_model_settings("openai_whisper_local")

    if not is_local_stt_available():
        # add an text to inform the user that local stt is not available
        information_text = QLabel(
            "Please install gpt-computer-assistant[local_stt] to use local STT"
        )
        settings_dialog.layout().addWidget(information_text)
        stt_model_select.setEnabled(False)

    stt_model_select.currentIndexChanged.connect(on_stt_model_change)

    # Add an separator
    separator = QLabel("------------------------------------------------")
    settings_dialog.layout().addWidget(separator)

    # Add an powered by label
    powered_by_label = QLabel("Powered by Upsonic <3")
    # Make label bold
    font = powered_by_label.font()
    font.setBold(True)
    powered_by_label.setFont(font)

    settings_dialog.layout().addWidget(powered_by_label)

    settings_dialog.exec_()
