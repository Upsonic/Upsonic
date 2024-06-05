import os

from dotenv import load_dotenv
load_dotenv(".env")

currently_dir  = os.path.dirname(os.path.abspath(__file__))

artifacts_dir = os.path.join(currently_dir, "artifacts")
media_dir = os.path.join(currently_dir, "media")

if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

mic_record_location = os.path.join(artifacts_dir, "mic_record.wav")

system_sound_location = os.path.join(artifacts_dir, "system_sound.wav")



just_screenshot_path = os.path.join(artifacts_dir, "screenshot.png")
screenshot_path = os.path.join(artifacts_dir, "screenshot_with_text.png")






the_profile = "default"

def set_profile(profile):
    print("Setting profile to", profile)
    global the_profile
    the_profile = profile
    
def get_profile():
    global the_profile
    return the_profile

def get_history_db():
    global the_profile
    return os.path.join(artifacts_dir, f"history_{the_profile}.db")

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
        


model_settings_db = os.path.join(artifacts_dir, "model_settings.db")

def save_model_settings(model):
    with open(model_settings_db, 'w') as f:
        f.write(model)

def load_model_settings():
    if not os.path.exists(model_settings_db):
        return "gpt-4o"
    with open(model_settings_db, 'r') as f:
        return f.read()




just_text_model = os.path.join(artifacts_dir, "just_text_model.db")

def activate_just_text_model():
    with open(just_text_model, 'w') as f:
        f.write("1")

def deactivate_just_text_model():
    with open(just_text_model, 'w') as f:
        f.write("0")

def is_just_text_model_active():
    if not os.path.exists(just_text_model):
        return False
    with open(just_text_model, 'r') as f:
        return f.read() == "1"
    


icon_16_path = os.path.join(media_dir, "icon_16.png")
icon_24_path = os.path.join(media_dir, "icon_24.png")
icon_32_path = os.path.join(media_dir, "icon_32.png")
icon_48_path = os.path.join(media_dir, "icon_48.png")
icon_256_path = os.path.join(media_dir, "icon_256.png")


screenshot_icon_path = os.path.join(media_dir, "Screenshot.png")
audio_icon_path = os.path.join(media_dir, "Audio.png")
microphone_icon_path = os.path.join(media_dir, "Microphone.png")




agents = []