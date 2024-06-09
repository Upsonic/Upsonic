try:
    from ..utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile, load_model_settings, save_model_settings, load_openai_url, load_groq_api_key
    from ..agent.chat_history import clear_chat_history
    from ..llm_settings import llm_show_name, llm_settings
except ImportError:
    from utils.db import screenshot_path, save_api_key, load_api_key, activate_just_text_model, deactivate_just_text_model, is_just_text_model_active, set_profile, get_profile, load_model_settings, save_model_settings, load_openai_url, load_groq_api_key
    from agent.chat_history import clear_chat_history
    from llm_settings import llm_show_name, llm_settings
    
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt5.QtWidgets import QComboBox, QLabel


def llmsettings_popup(self):
    from ..gpt_computer_assistant import the_input_box, the_main_window

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

        the_main_window.update_from_thread("Saved API Key")
        settings_dialog.close()

    save_button.clicked.connect(lambda: save_api_key_(api_key_input.text()))
    settings_dialog.layout().addWidget(save_button)

    # Start of new OpenAI Base URL settings
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
        settings_dialog.close()

    openai_url_save_button = QPushButton("Save URL")
    openai_url_save_button.clicked.connect(save_openai_url_)
    settings_dialog.layout().addWidget(openai_url_save_button)
    # End of new OpenAI Base URL settings

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
        settings_dialog.close()

    groq_save_button.clicked.connect(
        lambda: groq_save_api_key_(groq_api_key_input.text())
    )
    settings_dialog.layout().addWidget(groq_save_button)

    def hide_openai():
        api_key_label.hide()
        api_key_input.hide()
        openai_url_label.hide()
        openai_url_input.hide()
        save_button.hide()
        openai_url_save_button.hide()

    def hide_groq():
        groq_api_key_label.hide()
        groq_api_key_input.hide()
        groq_save_button.hide()

    def show_openai():
        api_key_label.show()
        api_key_input.show()
        openai_url_label.show()
        openai_url_input.show()
        save_button.show()
        openai_url_save_button.show()

    def show_groq():
        groq_api_key_label.show()
        groq_api_key_input.show()
        groq_save_button.show()

    hide_openai()
    hide_groq()

    print("LLLM SETTINGS", list(llm_show_name.keys()))

    # Add a select box with the options OpenAI and Olamo
    model_label = QLabel("Model")
    model_select = QComboBox()
    model_select.addItems(
        list(llm_show_name.keys())
    )

    settings_dialog.layout().addWidget(model_label)
    settings_dialog.layout().addWidget(model_select)

    # currently model
    current_model = load_model_settings()
    # lets set index of current model
    for i, model in enumerate(llm_show_name.keys()):
        if model == current_model:
            model_select.setCurrentIndex(i)



    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "openai":
        show_openai()

    if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "groq":
        show_groq()

    if not llm_settings[llm_show_name[model_select.currentText()]]["transcription"]:
        from ..gpt_computer_assistant import the_main_window

        the_main_window.remove_painting()

    if not llm_settings[llm_show_name[model_select.currentText()]]["vision"]:
        from ..gpt_computer_assistant import the_main_window

        the_main_window.remove_screenshot_button()




    def on_model_change():
        hide_openai()
        hide_groq()


        the_save_string = llm_show_name[model_select.currentText()]
        save_model_settings(the_save_string)





        if llm_settings[llm_show_name[model_select.currentText()]]["transcription"] == False: 
            from ..gpt_computer_assistant import the_main_window

            the_main_window.remove_painting()



        if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "openai":
            show_openai()
            openai_url_label.show()
            openai_url_input.show()
            openai_url_save_button.show()
            from ..gpt_computer_assistant import the_main_window

            the_main_window.activate_painting()

        if llm_settings[llm_show_name[model_select.currentText()]]["vision"]:
            the_main_window.add_screenshot_button()
        else:
            the_main_window.remove_screenshot_button()




        if llm_settings[llm_show_name[model_select.currentText()]]["provider"] == "groq":
            show_groq


    model_select.currentIndexChanged.connect(on_model_change)

    settings_dialog.exec_()
