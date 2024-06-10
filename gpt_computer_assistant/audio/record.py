try:
    from ..gui.signal import *
    from ..utils.db import mic_record_location, system_sound_location, load_user_id
    from ..utils.telemetry import my_tracer, os_name
except ImportError:
    from gui.signal import *
    from utils.db import mic_record_location, system_sound_location, load_user_id
    from utils.telemetry import my_tracer, os_name
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavfile
import soundcard as sc
import threading

samplerate = 48000  # Updated samplerate for better quality
channels = 1
recording = False

audio_data = None

user_id = load_user_id()
os_name_ = os_name()

the_input_box_pre = None

def start_recording(take_system_audio=False):
    """Start recording audio from microphone and/or system sound.

    Args:
        take_system_audio (bool, optional): Whether to record system sound. Defaults to False.
    """
    with my_tracer.start_span("start_recording") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)

        global the_input_box_pre
        from ..gpt_computer_assistant import the_input_box, the_main_window
        the_input_box_pre = the_input_box.toPlainText()

        the_main_window.update_from_thread("Click again when recording is done")
        global recording, audio_data
        recording = True
        audio_data = np.array([], dtype="float32")
        print("Recording started...")

        def callback(indata, frames, time, status):
            global audio_data
            if recording:
                audio_data = np.append(audio_data, indata)

    def record_audio():
        with my_tracer.start_span("record_audio") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("os_name", os_name_)
            global recording
            mics = sc.all_microphones(include_loopback=True)
            default_mic = mics[0]
            data = []
            with default_mic.recorder(samplerate=148000) as mic:
                print("Recording...")
                while recording:
                    frame = mic.record(numframes=4096)
                    data.append(frame)
            data = np.concatenate(data, axis=0)
            data_int16 = (data * 32767).astype("int16")
            wavfile.write(system_sound_location, 148000, data_int16)

    if take_system_audio:
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

    with sd.InputStream(callback=callback, channels=channels, samplerate=samplerate):
        while recording:
            sd.sleep(100)

    if not recording:
        sf.write(mic_record_location, audio_data, samplerate)
        print("Audio saved as voice_input.wav")
        signal_handler.recording_stopped.emit()

def stop_recording():
    """Stop recording audio."""
    global recording
    recording = False
    print("Recording stopped")
