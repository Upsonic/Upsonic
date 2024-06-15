from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from ..utils.db import *
from ..agent.chat_history import clear_chat_history

def settings_popup(self):
    """
    Display a settings popup dialog for configuring various options.

    This function creates a settings dialog with options to reset chat history, enable/disable the just text model,
    and change the active profile.

    Parameters:
    - self: Reference to the main application window.

    Returns:
    - None
    """
    from ..gpt_computer_assistant import the_input_box, the_main_window

    settings_dialog = QDialog()
    settings_dialog.setWindowTitle("Settings")
    settings_dialog.setWindowModality(Qt.ApplicationModal)

    settings_dialog.setLayout(QVBoxLayout())

    reset_memory_button = QPushButton("Reset Memory")

    def clear_chat_history_():
        """
        Clear the chat history and update the main window.

        This function clears the chat history and updates the main window with a notification.

        Returns:
        - None
        """
        clear_chat_history()
        the_main_window.update_from_thread("Cleared Chat History")
        settings_dialog.close()

    reset_memory_button.clicked.connect(clear_chat_history_)
    settings_dialog.layout().addWidget(reset_memory_button)

    just_text_button = QPushButton("Enable Just Text Model")

    settings_dialog.layout().addWidget(just_text_button)

    if is_just_text_model_active():
        just_text_button.setText("Disable Just Text Model")

        def deactivate_just_text_model_():
            """
            Deactivate the just text model and update the main window.

            This function deactivates the just text model and updates the main window with a notification.

            Returns:
            - None
            """
            deactivate_just_text_model()
            the_main_window.update_from_thread("Disabled Just Text Model")
            settings_dialog.close()

        just_text_button.clicked.connect(deactivate_just_text_model_)
    else:

        def activate_just_text_model_():
            """
            Activate the just text model and update the main window.

            This function activates the just text model and updates the main window with a notification.

            Returns:
            - None
            """
            activate_just_text_model()
            the_main_window.update_from_thread("Enabled Just Text Model")
            settings_dialog.close()

        just_text_button.clicked.connect(activate_just_text_model_)

    settings_dialog.layout().addWidget(QLabel("Profile"))
    profile_input = QLineEdit()

    profile_input.setText(get_profile())
    settings_dialog.layout().addWidget(profile_input)
    profile_save_button = QPushButton("Save")

    def set_profile_(profile):
        """
        Set the active profile and update the main window.

        This function sets the active profile based on user input and updates the main window with a notification.

        Parameters:
        - profile (str): The profile name to set.

        Returns:
        - None
        """
        set_profile(profile)
        the_main_window.update_from_thread("Saved Profile")
        settings_dialog.close()

    profile_save_button.clicked.connect(lambda: set_profile_(profile_input.text()))
    settings_dialog.layout().addWidget(profile_save_button)


    dark_mode_button = QPushButton("Enable Dark Mode")

    settings_dialog.layout().addWidget(dark_mode_button)

    if is_dark_mode_active():
        dark_mode_button.setText("Disable Dark Mode")

        def deactivate_dark_mode_():
            """
            Deactivate dark mode and update the main window.

            This function deactivates dark mode and updates the main window with a notification.

            Returns:
            - None
            """
            deactivate_dark_mode()
            the_main_window.update_from_thread("Disabled Dark Mode")
            the_main_window.light_mode()
            settings_dialog.close()

        dark_mode_button.clicked.connect(deactivate_dark_mode_)
    else:

            def activate_dark_mode_():
                """
                Activate dark mode and update the main window.
    
                This function activates dark mode and updates the main window with a notification.
    
                Returns:
                - None
                """
                activate_dark_mode()
                the_main_window.update_from_thread("Enabled Dark Mode")
                the_main_window.dark_mode()
                settings_dialog.close()
    
            dark_mode_button.clicked.connect(activate_dark_mode_)




    predefined_agents_button = QPushButton("Enable Predefined Agents (Good Results, Long Response Time)")

    settings_dialog.layout().addWidget(predefined_agents_button)

    try:
        import crewai
        if is_predefined_agents_setting_active():
            predefined_agents_button.setText("Disable Predefined Agents (Bad Results, Short Response Time)")

            def deactivate_predefined_agents_():
                deactivate_predefined_agents_setting()
                the_main_window.update_from_thread("Disabled Predefined Agents (Bad Results, Short Response Time)")
                settings_dialog.close()

            predefined_agents_button.clicked.connect(deactivate_predefined_agents_)
        else:
                
                def activate_predefined_agents_():
                    activate_predefined_agents_setting()
                    the_main_window.update_from_thread("Enabled Predefined Agents (Good Results, Long Response Time)")
                    settings_dialog.close()
        
                predefined_agents_button.clicked.connect(activate_predefined_agents_)

    except:
         predefined_agents_button.setText("Install gpt-computer-assistant[agentic]")






    online_tools_button = QPushButton("Enable Upsonic Tiger Tools - More Capability (Recommended)")

    settings_dialog.layout().addWidget(online_tools_button)

    if is_online_tools_setting_active():
        online_tools_button.setText("Disable Upsonic Tiger Tools - Low Capability (Not Recommended)")

        def deactivate_online_tools_():
            deactivate_online_tools_setting()
            the_main_window.update_from_thread("Disabled Upsonic Tiger Tools - Low Capability (Not Recommended)")
            settings_dialog.close()

        online_tools_button.clicked.connect(deactivate_online_tools_)
    else:
            
            def activate_online_tools_():
                activate_online_tools_setting()
                the_main_window.update_from_thread("Enabled Upsonic Tiger Tools - More Capability (Recommended)")
                settings_dialog.close()
    
            online_tools_button.clicked.connect(activate_online_tools_)





    auto_stop_recording_button = QPushButton("Enable Auto Stop Recording")

    settings_dialog.layout().addWidget(auto_stop_recording_button)

    if is_auto_stop_recording_setting_active():
        auto_stop_recording_button.setText("Disable Auto Stop Recording")

        def deactivate_auto_stop_recording_():
            deactivate_auto_stop_recording_setting()
            the_main_window.update_from_thread("Disabled Auto Stop Recording")
            settings_dialog.close()

        auto_stop_recording_button.clicked.connect(deactivate_auto_stop_recording_)
    else:
            
            def activate_auto_stop_recording_():
                activate_auto_stop_recording_setting()
                the_main_window.update_from_thread("Enabled Auto Stop Recording")
                settings_dialog.close()
    
            auto_stop_recording_button.clicked.connect(activate_auto_stop_recording_)





    api_key_label = QLabel("Wakeword - Pvporcupine API Key")
    settings_dialog.layout().addWidget(api_key_label)
    api_key_input = QLineEdit()
    api_key = load_pvporcupine_api_key()
    api_key_input.setText(api_key)
    settings_dialog.layout().addWidget(api_key_input)
    save_button = QPushButton("Save")

    def save_api_key_(api_key):
        first_time = True
        if api_key != "CHANGE_ME":
            first_time = False
        save_pvporcupine_api_key(api_key)

        the_main_window.update_from_thread("Wake word activated, just say 'Her Computer' or jarvis to activate the assistant")
        if first_time:
            the_main_window.wake_word_trigger()
        settings_dialog.close()

    save_button.clicked.connect(lambda: save_api_key_(api_key_input.text()))
    settings_dialog.layout().addWidget(save_button)





    wake_word_button = QPushButton("Enable Wake Word")

    settings_dialog.layout().addWidget(wake_word_button)

    missing_parts = False
    try:
         import pyaudio
    except:
        missing_parts = True


    if api_key == "CHANGE_ME":
        wake_word_button.setText("Please Set Pvporcupine API Key First")
    elif missing_parts:
        wake_word_button.setText("Please Install gpt-computer-assistant[wakeword]")
    else:

        if is_wake_word_active():
            wake_word_button.setText("Disable Wake Word")

            def deactivate_wake_word_():
                deactivate_wake_word()
                the_main_window.update_from_thread("Disabled Wake Word")
                the_main_window.wake_word_active = False
                settings_dialog.close()

            wake_word_button.clicked.connect(deactivate_wake_word_)
        else:
                
                def activate_wake_word_():
                    activate_wake_word()
                    the_main_window.update_from_thread("Enabled Wake Word")
                    the_main_window.wake_word_active = True
                    the_main_window.wake_word_trigger()
                    settings_dialog.close()
        
                wake_word_button.clicked.connect(activate_wake_word_)




    settings_dialog.exec_()
