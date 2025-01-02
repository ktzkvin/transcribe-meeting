import os
from uuid import uuid4

from huggingface_hub import login
from dotenv import load_dotenv
from rx.core import Observer

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.models import SegmentationModel, EmbeddingModel
from diart.sources import MicrophoneAudioSource, FileAudioSource, AudioSource
from src.schema.config.diarization import AudioConfig, load_config_from_json
from src.producer.diarization import JsonStdoutProducer

load_dotenv()

login(token=os.environ["HF_TOKEN"])


def main(conf_path: str = "config/diarization.json", _id: str = str(uuid4())):
    audio_config = load_config_from_json(json_file_path=conf_path)

    segmentation = SegmentationModel.from_pyannote(
        audio_config.segmentation_model.value
    )
    embedding = EmbeddingModel.from_pyannote(audio_config.embedding_model.value)
    observers = JsonStdoutProducer(_id=_id)

    if audio_config.audio_path:
        audio = FileAudioSource(
            file=audio_config.audio_path, sample_rate=audio_config.sample_rate
        )
    else:
        audio = MicrophoneAudioSource(device=audio_config.mic)
    config = SpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        duration=audio_config.duration,
        step=audio_config.step,
        latency=audio_config.latency,
        tau_active=audio_config.tau_active,
        rho_update=audio_config.rho_update,
        delta_new=audio_config.delta_new,
        gamma=audio_config.gamma,
        beta=audio_config.beta,
        max_speaker=audio_config.max_speakers,
        normalize_embedding_weights=audio_config.normalize_embedding_weights,
        device=audio_config.device if audio_config.device else "cpu",
    )

    pipeline = SpeakerDiarization(config=config)
    inference = StreamingInference(pipeline, audio, do_plot=False)
    inference.attach_observers(observers)
    inference()


if __name__ == "__main__":
    main()
