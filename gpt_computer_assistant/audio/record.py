try:
    from ..gui.signal import *
    from ..utils.db import *
    from ..utils.telemetry import my_tracer, os_name
except ImportError:
    from gui.signal import *
    from utils.db import *
    from utils.telemetry import my_tracer, os_name
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavfile
import soundcard as sc
import threading
import time

samplerate = 48000  # Updated samplerate for better quality
channels = 1
recording = False

audio_data = None

user_id = load_user_id()
os_name_ = os_name()

the_input_box_pre = None



import queue

# Initialize a queue to keep the last N audio levels (rolling window)
audio_levels = queue.Queue(maxsize=10)  # Adjust size as needed

def calculate_dynamic_threshold():
    """Calculate a dynamic threshold based on recent audio levels."""
    if audio_levels.qsize() == 0:
        return 0.01  # Default threshold if no data is available
    else:
        # Calculate the average of the last N audio levels
        return np.mean(list(audio_levels.queue)) * 2  # Adjust multiplier as needed


silence_start_time = None

auto_stop_recording = True


def start_recording(take_system_audio, buttonhandler):
    """Start recording audio from microphone and/or system sound.


    """
    with my_tracer.start_span("start_recording") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)

        global the_input_box_pre
        from ..gpt_computer_assistant import the_input_box, the_main_window
        the_input_box_pre = the_input_box.toPlainText()

        the_main_window.update_from_thread("Click again when recording is done")
        global recording, audio_data, silence_start_time, auto_stop_recording
        recording = True
        audio_data = np.array([], dtype="float32")
        print("Recording started...")

        threshold = 0.01  # Define the threshold for stopping the recording
        silence_duration = 2  # Duration in seconds to consider as silence before stopping
        silence_start_time = None
        recording_start_time = time.time()  # Record the start time of the recording

        auto_stop_recording = is_auto_stop_recording_setting_active()


        def callback(indata, frames, time_info, status):
            global audio_data, recording, silence_start_time, auto_stop_recording
            current_level = np.max(np.abs(indata))

            
            # Add the current level to the queue
            if audio_levels.full():
                audio_levels.get()  # Remove the oldest level if the queue is full
            audio_levels.put(current_level)
            
            # Calculate dynamic threshold based on recent audio levels
            dynamic_threshold = calculate_dynamic_threshold()


            if recording:
                audio_data = np.append(audio_data, indata)
                # Check if the audio is below the dynamic threshold
                if current_level < dynamic_threshold and auto_stop_recording:
                    if silence_start_time is None:
                        silence_start_time = time.time()  # Mark the start of silence

                    # Ensure recording has been ongoing for at least 3 seconds before considering auto-stop
                    elif (time.time() - silence_start_time) > silence_duration and (time.time() - recording_start_time) > 3:
                        recording = False
                        buttonhandler.recording = False

                else:
                    silence_start_time = None


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
