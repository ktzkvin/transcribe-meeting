import argparse
import logging
import os
import traceback

from huggingface_hub import login
from dotenv import load_dotenv

from rx.core import Observer
import rx.operators as ops
from diart import SpeakerDiarizationConfig
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding
from diart.models import SegmentationModel, EmbeddingModel

from src.sources.audio_kafka import KafkaAudioSource
from src.schema.config.audio import KafkaAudioConfig
from src.observers.logger import DebugAudioMetadataLogger, DebugLogger
from src.diarization.diart import KafkaDiarization

load_dotenv()

login(token=os.environ["HF_TOKEN"])
import os
from uuid import uuid4

from huggingface_hub import login
from dotenv import load_dotenv
from rx.core import Observer

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.models import SegmentationModel, EmbeddingModel

from src.schema.config.diarization import load_config_from_json

load_dotenv()

login(token=os.environ["HF_TOKEN"])


def main(
    audio: KafkaAudioSource,
    observer_audio: Observer,
    conf_path: str = "config/diarization.json",
    _id: str = str(uuid4()),
):
    audio_config = load_config_from_json(json_file_path=conf_path)

    segmentation = SegmentationModel.from_pyannote(
        audio_config.segmentation_model.value
    )
    embedding = EmbeddingModel.from_pyannote(audio_config.embedding_model.value)

    config = SpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        duration=15,
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

    pipeline = KafkaDiarization(config=config)
    audio.stream.pipe(ops.do(observer_audio), ops.map(pipeline)).subscribe(
        on_error=lambda _: traceback.print_exc()
    )
    print(79 * "*")
    audio.read()


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Kafka Audio Consumer with Debugging and Transcription"
    )

    # Kafka settings
    parser.add_argument(
        "--topic",
        type=str,
        default="audio",
        help="Kafka topic to consume from (default: 'audio')",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:29092",
        help="Kafka bootstrap servers (default: 'localhost:29092')",
    )

    # Audio settings
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Size of each audio chunk (default: 5)",
    )

    # Logging settings
    parser.add_argument(
        "--log-file",
        type=str,
        default="consumer-diarization-debug.log",
        help="Log file for saving debug output (default: 'consumer-diarization-debug.log')",
    )
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="Whether to log audio data (default: False)",
    )

    # Transcription settings
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use for transcription (default: 'small')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Whisper transcription (default: 'cuda')",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up the Kafka audio source
    source_audio = KafkaAudioSource(
        kafka_config=KafkaAudioConfig(
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            groud_id="diarization-consumer-audio",
            enable_auto_commit=False,
        ),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )

    # Set up the logger
    logger = logging.getLogger("consumer-diarization-debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.log_file))

    # Create the observer with the provided logging settings
    observer = DebugAudioMetadataLogger(logger=logger, write_data=args.write_data)

    # Run the main function
    main(audio=source_audio, observer_audio=observer)
