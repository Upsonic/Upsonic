import os
import sys
from PyQt5.QtWidgets import QApplication

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
        import crewai
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
        from .utils.db import load_tts_model_settings, load_stt_model_settings
    except ImportError:
        from utils.db import load_tts_model_settings, load_stt_model_settings

    if load_tts_model_settings() != "openai":
        from .audio.tts_providers.microsoft_local import preload_tts_microsoft_local
        preload_tts_microsoft_local()
    
    if load_stt_model_settings() != "openai":
        from .audio.stt_providers.openai_whisper_local import preload_stt_openai_whisper_local
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
    sys.exit(app.exec_())
