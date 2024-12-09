import os
import uuid
from dotenv import load_dotenv

try:
    from .folder import currently_dir, artifacts_dir, media_dir
    from .kot_db import kot_db_
except:
    from folder import currently_dir, artifacts_dir, media_dir
    from kot_db import kot_db_


load_dotenv(".env")



if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)

mic_record_location = os.path.join(artifacts_dir, "mic_record.wav")
system_sound_location = os.path.join(artifacts_dir, "system_sound.wav")
just_screenshot_path = os.path.join(artifacts_dir, "screenshot.png")
screenshot_path = os.path.join(artifacts_dir, "screenshot_with_text.png")
the_profile = "default"


def set_profile(profile):
    """Set the active profile."""
    print("Setting profile to", profile)
    global the_profile
    the_profile = profile


def get_profile():
    """Get the active profile."""
    global the_profile
    return the_profile


def get_history_db():
    """Get the history database path based on the active profile."""
    global the_profile
    return os.path.join(artifacts_dir, f"history_{the_profile}.db")



# API KEY SAVING AND LOADING
def save_api_key(api_key):
    kot_db_.set("openai_api_key", api_key)
def load_api_key():
    if kot_db_.get("openai_api_key"):
        return kot_db_.get("openai_api_key")
    else:
        env_variable = os.getenv("OPENAI_API_KEY")
        if env_variable:
            save_api_key(env_variable)
            return env_variable
        return "CHANGE_ME"

def save_anthropic_api_key(api_key):
    kot_db_.set("anthropic_api_key", api_key)
def load_anthropic_api_key():
    if kot_db_.get("anthropic_api_key"):
        return kot_db_.get("anthropic_api_key")
    else:
        env_variable = os.getenv("ANTHROPIC_API_KEY")
        if env_variable:
            save_api_key(env_variable)
            return env_variable
        return "CHANGE_ME"


# OPENAI URL SAVING AND LOADING
def save_openai_url(url):
    kot_db_.set("openai_url", url)
def load_openai_url():
    if kot_db_.get("openai_url"):
        return kot_db_.get("openai_url")
    else:
        return "default"


# API VERSION SAVING AND LOADING
def save_api_version(url):
    kot_db_.set("api_version", url)
def load_api_version():
    if kot_db_.get("api_version"):
        return kot_db_.get("api_version")
    else:
        return "CHANGE_ME"


model_settings_db = os.path.join(artifacts_dir, "model_settings.db")


def save_model_settings(model):
    """Save the model settings to a file."""
    with open(model_settings_db, "w") as f:
        f.write(model)


def load_model_settings():
    """Load the model settings from a file."""
    if not os.path.exists(model_settings_db):
        return "gpt-4o"
    with open(model_settings_db, "r") as f:
        return f.read()


just_text_model = os.path.join(artifacts_dir, "just_text_model.db")


def activate_just_text_model():
    """Activate the just text model."""
    with open(just_text_model, "w") as f:
        f.write("1")


def deactivate_just_text_model():
    """Deactivate the just text model."""
    with open(just_text_model, "w") as f:
        f.write("0")


def is_just_text_model_active():
    """Check if the just text model is active."""
    if not os.path.exists(just_text_model):
        return False
    with open(just_text_model, "r") as f:
        return f.read() == "1"


# Define paths for icons and other media
icon_16_path = os.path.join(media_dir, "icon_16.png")
icon_24_path = os.path.join(media_dir, "icon_24.png")
icon_32_path = os.path.join(media_dir, "icon_32.png")
icon_48_path = os.path.join(media_dir, "icon_48.png")
icon_48_active_path = os.path.join(media_dir, "icon_48_active.png")
icon_256_path = os.path.join(media_dir, "icon_256.png")
screenshot_icon_path = os.path.join(media_dir, "Screenshot.png")
audio_icon_path = os.path.join(media_dir, "Audio.png")
microphone_icon_path = os.path.join(media_dir, "Microphone.png")
up_icon_path = os.path.join(media_dir, "Up.png")
down_icon_path = os.path.join(media_dir, "Down.png")
double_down_icon_path = os.path.join(media_dir, "Double_down.png")

click_sound_path = os.path.join(media_dir, "boop.mp3")

gca_logo_path = os.path.join(media_dir, "gca_logo.png")

agents = []  # Placeholder for agents data

groqkey = os.path.join(artifacts_dir, "groqkey.db")


def save_groq_api_key(api_key):
    """Save the Groq API key to a file."""
    with open(groqkey, "w") as f:
        f.write(api_key)


def load_groq_api_key():
    """Load the Groq API key from a file or environment variables."""
    if not os.path.exists(groqkey):
        env = os.getenv("GROQ_API_KEY")
        if env:
            save_api_key(env)
            return env
        else:
            return "CHANGE_ME"
    with open(groqkey, "r") as f:
        return f.read()


user_id_db = os.path.join(artifacts_dir, "user_id.db")


def save_user_id():
    """Save a unique user ID to a file."""
    with open(user_id_db, "w") as f:
        uuid4 = str(uuid.uuid4())
        f.write(uuid4)
        return uuid4


def load_user_id():
    """Load the unique user ID from a file."""
    if not os.path.exists(user_id_db):
        return save_user_id()
    with open(user_id_db, "r") as f:
        return f.read()


collapse_setting = os.path.join(artifacts_dir, "collapse_setting.db")


def activate_collapse_setting():
    """Activate the collapse setting."""
    with open(collapse_setting, "w") as f:
        f.write("1")


def deactivate_collapse_setting():
    """Deactivate the collapse setting."""
    with open(collapse_setting, "w") as f:
        f.write("0")


def is_collapse_setting_active():
    """Check if the collapse setting is active."""
    if not os.path.exists(collapse_setting):
        return False
    with open(collapse_setting, "r") as f:
        return f.read() == "1"


# Define font directory path
font_dir = os.path.join(media_dir, "SF-Pro-Text-Bold.otf")


style_setting = os.path.join(artifacts_dir, "style_setting.db")


def activate_dark_mode():
    """Activate the dark mode setting."""
    with open(style_setting, "w") as f:
        f.write("1")


def deactivate_dark_mode():
    """Deactivate the dark mode setting."""
    with open(style_setting, "w") as f:
        f.write("0")


def is_dark_mode_active():
    """Check if the dark mode setting is active."""
    if not os.path.exists(style_setting):
        return True
    with open(style_setting, "r") as f:
        return f.read() == "1"


googlekey = os.path.join(artifacts_dir, "googlekey.db")


def save_google_api_key(api_key):
    """Save the Google Generative AI API key to a file."""
    with open(googlekey, "w") as f:
        f.write(api_key)


def load_google_api_key():
    """Load the Google Generative AI API key from a file or environment variables."""
    if not os.path.exists(googlekey):
        env = os.getenv("GOOGLE_API_KEY")
        if env:
            save_api_key(env)
            return env
        else:
            return "CHANGE_ME"
    with open(googlekey, "r") as f:
        return f.read()


predefined_agents_setting = os.path.join(artifacts_dir, "predefined_agents_setting.db")


def activate_predefined_agents_setting():
    """Activate the predefined agents setting setting."""
    with open(predefined_agents_setting, "w") as f:
        f.write("1")


def deactivate_predefined_agents_setting():
    """Deactivate the predefined agents setting setting."""
    with open(predefined_agents_setting, "w") as f:
        f.write("0")


def is_predefined_agents_setting_active():
    """Check if the predefined agents setting setting is active."""
    if not os.path.exists(predefined_agents_setting):
        return True
    with open(predefined_agents_setting, "r") as f:
        return f.read() == "1"


online_tools_setting = os.path.join(artifacts_dir, "online_tools.db")


def activate_online_tools_setting():
    """Activate the online_tools setting."""
    with open(online_tools_setting, "w") as f:
        f.write("1")


def deactivate_online_tools_setting():
    """Deactivate the online_tools setting."""
    with open(online_tools_setting, "w") as f:
        f.write("0")


def is_online_tools_setting_active():
    """Check if the online_tools setting is active."""
    if not os.path.exists(online_tools_setting):
        return False
    with open(online_tools_setting, "r") as f:
        return f.read() == "1"


auto_stop_recording_setting = os.path.join(artifacts_dir, "auto_stop_recording.db")


def activate_auto_stop_recording_setting():
    """Activate the auto_stop_recording setting."""
    with open(auto_stop_recording_setting, "w") as f:
        f.write("1")


def deactivate_auto_stop_recording_setting():
    """Deactivate the auto_stop_recording setting."""
    with open(auto_stop_recording_setting, "w") as f:
        f.write("0")


def is_auto_stop_recording_setting_active():
    """Check if the auto_stop_recording setting is active."""
    if not os.path.exists(auto_stop_recording_setting):
        return True
    with open(auto_stop_recording_setting, "r") as f:
        return f.read() == "1"


pvporcupine_api_key = os.path.join(artifacts_dir, "pvporcupine_api_key.db")


def save_pvporcupine_api_key(api_key):
    """Save the Pvporcupine AI API key to a file."""
    with open(pvporcupine_api_key, "w") as f:
        f.write(api_key)


def load_pvporcupine_api_key():
    """Load the Pvporcupine AI API key from a file or environment variables."""
    if not os.path.exists(pvporcupine_api_key):
        return "CHANGE_ME"
    with open(pvporcupine_api_key, "r") as f:
        return f.read()


wake_word_setting = os.path.join(artifacts_dir, "wake_word_setting.db")


def activate_wake_word():
    """Activate the wake_word_setting setting."""
    with open(wake_word_setting, "w") as f:
        f.write("1")


def deactivate_wake_word():
    """Deactivate the wake_word_setting setting."""
    with open(wake_word_setting, "w") as f:
        f.write("0")


def is_wake_word_active():
    """Check if the wake_word_setting setting is active."""
    try:
        import pyaudio
    except ImportError:
        return False
    if not os.path.exists(wake_word_setting):
        return True
    with open(wake_word_setting, "r") as f:
        return f.read() == "1"


wake_word_screen_setting = os.path.join(artifacts_dir, "wake_word_screen_setting.db")


def activate_wake_word_screen_setting():
    """Activate the wake_word_screen setting."""
    with open(wake_word_screen_setting, "w") as f:
        f.write("1")


def deactivate_wake_word_screen_setting():
    """Deactivate the wake_word_screen setting."""
    with open(wake_word_screen_setting, "w") as f:
        f.write("0")


def is_wake_word_screen_setting_active():
    """Check if the wake_word_screen setting is active."""
    if not os.path.exists(wake_word_screen_setting):
        return True
    with open(wake_word_screen_setting, "r") as f:
        return f.read() == "1"


continuously_conversations_setting = os.path.join(
    artifacts_dir, "continuously_conversations_setting.db"
)


def activate_continuously_conversations_setting():
    """Activate the continuously_conversations setting."""
    with open(continuously_conversations_setting, "w") as f:
        f.write("1")


def deactivate_continuously_conversations_setting():
    """Deactivate the continuously_conversations setting."""
    with open(continuously_conversations_setting, "w") as f:
        f.write("0")


def is_continuously_conversations_setting_active():
    """Check if the continuously_conversations setting is active."""
    if not os.path.exists(continuously_conversations_setting):
        return False
    with open(continuously_conversations_setting, "r") as f:
        return f.read() == "1"


tts_model_settings_db = os.path.join(artifacts_dir, "tts_model_settings.db")


def save_tts_model_settings(model):
    """Save the tts model settings to a file."""
    with open(tts_model_settings_db, "w") as f:
        f.write(model)


def load_tts_model_settings():
    """Load the tts model settings from a file."""
    if not os.path.exists(tts_model_settings_db):
        return "openai"
    with open(tts_model_settings_db, "r") as f:
        return f.read()


stt_model_settings_db = os.path.join(artifacts_dir, "stt_model_settings.db")


def save_stt_model_settings(model):
    """Save the stt model settings to a file."""
    with open(stt_model_settings_db, "w") as f:
        f.write(model)


def load_stt_model_settings():
    """Load the stt model settings from a file."""
    if not os.path.exists(stt_model_settings_db):
        return "openai"
    with open(stt_model_settings_db, "r") as f:
        return f.read()


logo_active_setting = os.path.join(artifacts_dir, "logo_active_setting.db")


def activate_logo_active_setting():
    """Activate the logo_active_setting."""
    with open(logo_active_setting, "w") as f:
        f.write("1")


def deactivate_logo_active_setting():
    """Deactivate the logo_active_setting."""
    with open(logo_active_setting, "w") as f:
        f.write("0")


def is_logo_active_setting_active():
    """Check if the logo_active_setting is active."""
    if not os.path.exists(logo_active_setting):
        return False
    with open(logo_active_setting, "r") as f:
        return f.read() == "1"


logo_file_path = os.path.join(artifacts_dir, "loog_file.db")


def save_logo_file_path(model):
    """Save the logo_file_path to a file."""
    with open(logo_file_path, "w") as f:
        f.write(model)


def load_logo_file_path():
    """Load the logo_file_path from a file."""
    if not os.path.exists(logo_file_path):
        return icon_256_path
    with open(logo_file_path, "r") as f:
        return f.read()


custom_logo_path = os.path.join(artifacts_dir, "custom_logo_path.png")


long_gca_setting = os.path.join(artifacts_dir, "long_gca_setting.db")


def activate_long_gca_setting():
    """Activate the long_gca_setting."""
    with open(long_gca_setting, "w") as f:
        f.write("1")


def deactivate_long_gca_setting():
    """Deactivate the long_gca_setting."""
    with open(long_gca_setting, "w") as f:
        f.write("0")


def is_long_gca_setting_active():
    """Check if the long_gca_setting is active."""
    if not os.path.exists(long_gca_setting):
        return True
    with open(long_gca_setting, "r") as f:
        return f.read() == "1"


location_setting = os.path.join(artifacts_dir, "location_setting.db")


def save_location_setting(model):
    """Save the location_setting to a file."""
    with open(location_setting, "w") as f:
        f.write(model)


def load_location_setting():
    """Load the location_setting from a file."""
    if not os.path.exists(location_setting):
        return "right"
    with open(location_setting, "r") as f:
        return f.read()
