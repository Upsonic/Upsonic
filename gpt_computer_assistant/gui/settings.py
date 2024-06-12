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
        just_text_button.setText("Disabled Just Text Model")

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

    if is_predefined_agents_setting_active():
        predefined_agents_button.setText("Disable Predefined Agents (Bad Results, Short Response Time)")

        def deactivate_predefined_agents_():
            """
            Deactivate dark mode and update the main window.

            This function deactivates dark mode and updates the main window with a notification.

            Returns:
            - None
            """
            deactivate_predefined_agents_setting()
            the_main_window.update_from_thread("Disabled Predefined Agents (Bad Results, Short Response Time)")
            the_main_window.light_mode()
            settings_dialog.close()

        predefined_agents_button.clicked.connect(deactivate_predefined_agents_)
    else:
            
            def activate_predefined_agents_():
                """
                Activate dark mode and update the main window.
    
                This function activates dark mode and updates the main window with a notification.
    
                Returns:
                - None
                """
                activate_predefined_agents_setting()
                the_main_window.update_from_thread("Enable Predefined Agents (Good Results, Long Response Time)")
                settings_dialog.close()
    
            predefined_agents_button.clicked.connect(activate_predefined_agents_)






    settings_dialog.exec_()
