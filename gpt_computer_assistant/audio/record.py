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
sound_data = None

user_id = load_user_id()
os_name_ = os_name()

def start_recording(take_system_audio=False):
    with my_tracer.start_span("start_recording") as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("os_name", os_name_)
        
        from ..gpt_computer_assistant import the_input_box
        the_input_box.setText("Click again when recording is done")
        global recording, audio_data
        recording = True
        audio_data = np.array([], dtype='float32')
        sound_data = np.array([], dtype='float32')
        print("Recording started...")

        def callback(indata, frames, time, status):
            global audio_data
            if recording:
                audio_data = np.append(audio_data, indata)
    






    def record_audio():
        with my_tracer.start_span("record_audio") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("os_name", os_name_)
            global recording, sound_data
            mics = sc.all_microphones(include_loopback=True)

            default_mic = mics[0]

            data = []

            with default_mic.recorder(samplerate=148000) as mic:
                print("Recording...")
                
                while recording:
                    frame = mic.record(numframes=4096)
                    data.append(frame)

            # Convert list to numpy array
            data = np.concatenate(data, axis=0)

            # Convert the recorded data to 16-bit PCM format
            data_int16 = (data * 32767).astype('int16')

            # Save the recorded data as a WAV file
            
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
    global recording
    recording = False
    print("Recording stopped")