import os

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
    args = parser.parse_args()
    profile = args.profile
    api_arg = args.api
    print("Profile:", profile)

    if profile is not None:
        from .utils.db import set_profile
        set_profile(profile)


        

    try:
        from .gpt_computer_assistant import QApplication, MainWindow, sys
    except ImportError:
        from gpt_computer_assistant import QApplication, MainWindow, sys
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    if api or api_arg:
        print("API Enabled")
        MainWindow.api_enabled = True
    
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
