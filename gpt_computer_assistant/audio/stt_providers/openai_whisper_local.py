model_ = None


def model():
    global model_
    if model_ is None:
        import whisper

        model_ = whisper.load_model("tiny")
    return model_


def preload_stt_openai_whisper_local():
    model()


def stt_openai_whisper_local(audio_file):
    result = model().transcribe(audio_file)
    return result["text"]
