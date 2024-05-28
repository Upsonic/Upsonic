


def start():

    # get --profile argument with library
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="profile to use")
    args = parser.parse_args()
    profile = args.profile
    print("Profile:", profile)

    if profile != None:
        from .utils.db import set_profile
        set_profile(profile)
    

    from .gpt_computer_assistant import QApplication, MainWindow, sys
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
