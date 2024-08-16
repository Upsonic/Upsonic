import os
import platform
import sys
import webbrowser

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from pynput import keyboard


def start(api=False):
    """
    Starts the computer assistant application.

    This function starts the computer assistant application, which includes parsing command-line arguments
    to set the profile, initializing the graphical user interface, and starting the application event loop.

    Command-line Arguments:
    --profile (str): The profile to use for the application.

    Raises:
    ImportError: If the required modules or packages are not found.

    Returns:
    None
    """

    try:
        pass
    except:
        pass

    # get --profile argument with library
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="profile to use")
    parser.add_argument("--api", help="Enable API mode", action="store_true")

    parser.add_argument("--set_tts_provider", help="Set tts provider only")
    parser.add_argument("--set_stt_provider", help="Set stt provider only")

    parser.add_argument("--set_llm", help="Set llm model only")

    args = parser.parse_args()

    set_tts_provider = args.set_tts_provider

    if set_tts_provider is not None:
        from .utils.db import save_tts_model_settings

        save_tts_model_settings(set_tts_provider)
        return

    set_stt_provider = args.set_stt_provider

    if set_stt_provider is not None:
        from .utils.db import save_stt_model_settings

        save_stt_model_settings(set_stt_provider)
        return

    set_llm = args.set_llm

    if set_llm is not None:
        from .utils.db import save_model_settings

        save_model_settings(set_llm)
        return

    profile = args.profile

    api_arg = args.api
    print("Profile:", profile)

    if profile is not None:
        from .utils.db import set_profile

        set_profile(profile)

    try:
        from .utils.db import (
            load_tts_model_settings,
            load_stt_model_settings,
            is_logo_active_setting_active,
            load_logo_file_path,
        )
    except ImportError:
        from utils.db import (
            load_tts_model_settings,
            load_stt_model_settings,
            load_logo_file_path,
        )

    if load_tts_model_settings() != "openai":
        from .audio.tts_providers.microsoft_local import preload_tts_microsoft_local

        preload_tts_microsoft_local()

    if load_stt_model_settings() != "openai":
        from .audio.stt_providers.openai_whisper_local import (
            preload_stt_openai_whisper_local,
        )

        preload_stt_openai_whisper_local()

    try:
        from .gpt_computer_assistant import MainWindow
    except ImportError:
        from gpt_computer_assistant import MainWindow
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    if api or api_arg:
        print("API Enabled")
        MainWindow.api_enabled = True

    app = QApplication(sys.argv)
    ex = MainWindow()
    from PyQt5 import QtGui
    from PyQt5 import QtCore

    app_icon = QtGui.QIcon()

    app_icon.addFile(load_logo_file_path(), QtCore.QSize(48, 48))
    app.setWindowIcon(app_icon)

    ex.the_app = app

    # Create the tray
    menu_icon = QtGui.QIcon()
    menu_icon.addFile(load_logo_file_path(), QtCore.QSize(48, 48))

    menu_active_icon = QtGui.QIcon()
    menu_active_icon.addFile(load_logo_file_path(), QtCore.QSize(48, 48))

    tray = QSystemTrayIcon()
    tray.setIcon(menu_icon)
    tray.setVisible(True)
    ex.tray = tray
    ex.tray_active_icon = menu_active_icon
    ex.tray_icon = menu_icon

    # Create the menu
    menu = QMenu()

    ex.the_tray = tray

    show_menu = QAction("Show")

    def show_menu_connect():
        ex.setWindowState(Qt.WindowNoState)

    show_menu.triggered.connect(show_menu_connect)
    menu.addAction(show_menu)

    hide_menu = QAction("Hide")
    hide_menu.triggered.connect(ex.showMinimized)
    menu.addAction(hide_menu)

    menu.addSeparator()

    if platform.system() == "Darwin":
        the_text_of_screenshot_and_microphone = (
            "Action: ⌃+⌥+⌘+up Screenshot and Microphone"
        )
    else:
        the_text_of_screenshot_and_microphone = (
            "Action: ctrl+alt+windows+up Screenshot and Microphone"
        )
    screenshot_and_microphone = QAction(the_text_of_screenshot_and_microphone)

    def screenshot_and_microphone_connect():
        ex.setWindowState(Qt.WindowNoState)
        ex.screenshot_and_microphone_button_action()

    screenshot_listener = keyboard.GlobalHotKeys(
        {"<ctrl>+<alt>+<cmd>+<up>": screenshot_and_microphone_connect}
    )
    screenshot_listener.start()

    screenshot_and_microphone.triggered.connect(screenshot_and_microphone_connect)
    menu.addAction(screenshot_and_microphone)

    menu.addSeparator()

    action = QAction("Open GitHub Issues")
    action.triggered.connect(
        lambda: webbrowser.open(
            "https://github.com/onuratakan/gpt-computer-assistant/issues"
        )
    )
    menu.addAction(action)

    # Add a Quit option to the menu.
    quit = QAction("Quit")
    quit.triggered.connect(app.quit)
    menu.addAction(quit)

    # Add the menu to the tray
    tray.setContextMenu(menu)

    sys.exit(app.exec_())
