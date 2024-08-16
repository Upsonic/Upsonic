import soundfile as sf


synthesiser_ = None


def synthesiser():
    from transformers import pipeline

    global synthesiser_
    if synthesiser_ is None:
        synthesiser_ = pipeline("text-to-speech", "microsoft/speecht5_tts")
    return synthesiser_


embeddings_dataset_ = None


def embeddings_dataset():
    from datasets import load_dataset

    global embeddings_dataset_
    if embeddings_dataset_ is None:
        embeddings_dataset_ = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
    return embeddings_dataset_


speaker_embedding_ = None


def speaker_embedding():
    import torch

    global speaker_embedding_
    if speaker_embedding_ is None:
        speaker_embedding_ = torch.tensor(
            embeddings_dataset()[7306]["xvector"]
        ).unsqueeze(0)
    return speaker_embedding_


def preload_tts_microsoft_local():
    synthesiser()
    embeddings_dataset()
    speaker_embedding()


def tts_microsoft_local(text_chunk, location):
    speech = synthesiser()(
        text_chunk, forward_params={"speaker_embeddings": speaker_embedding()}
    )
    sf.write(location, speech["audio"], samplerate=speech["sampling_rate"])
    return location
