import os

from dotenv import load_dotenv
load_dotenv(".env")

currently_dir  = os.path.dirname(os.path.abspath(__file__))

artifacts_dir = os.path.join(currently_dir, "artifacts")

if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

mic_record_location = os.path.join(artifacts_dir, "mic_record.wav")

system_sound_location = os.path.join(artifacts_dir, "system_sound.wav")



just_screenshot_path = os.path.join(artifacts_dir, "screenshot.png")
screenshot_path = os.path.join(artifacts_dir, "screenshot_with_text.png")




history_db = os.path.join(artifacts_dir, "history.db")

openaikey = os.path.join(artifacts_dir, "openaikey.db")


def save_api_key(api_key):
        with open(openaikey, 'w') as f:
            f.write(api_key)

def load_api_key():
        if not os.path.exists(openaikey):
            env = os.getenv("OPENAI_API_KEY")
            if env:
                save_api_key(env)
                return env
            else:
                return "CHANGE_ME"
        with open(openaikey, 'r') as f:
            return f.read()