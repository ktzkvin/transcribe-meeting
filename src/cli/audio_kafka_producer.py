import argparse
import traceback
from typing import Union
import rx.operators as ops
from rx.core import Observer

from src.sources.audio import MicrophoneAudioSourceTimed, FileAudioSourceTimed
from src.producer.base import BaseProducer
from src.producer.kafka import KafkaDataclassProducer
from src.schema.config.audio import KafkaAudioConfig
from src.observers.logger import DebugLogger
import logging


def main(
    producer: BaseProducer,
    source_audio: Union[MicrophoneAudioSourceTimed, FileAudioSourceTimed],
    observer: Observer,
):
    source_audio.stream.pipe(ops.do(producer), ops.do(observer)).subscribe(
        on_error=lambda _: traceback.print_exc()
    )
    source_audio.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Kafka Producer Script")
    parser.add_argument(
        "--topic",
        type=str,
        default="audio",
        help="Name of the Kafka topic (default: 'audio')",
    )
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default="localhost:29092",
        help="Kafka bootstrap servers (default: 'localhost:29092')",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["microphone", "file"],
        default="microphone",
        help="Audio source: 'microphone' or 'file' (default: 'microphone')",
    )
    parser.add_argument(
        "--block-duration",
        type=int,
        default=2,
        help="Block duration in seconds for audio reading (default: 2)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone device ID to use (default: first available)",
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Path to the audio file if source='file'",
    )
    args = parser.parse_args()
    kafka_config = KafkaAudioConfig(
        topic=args.topic, bootstrap_servers=args.bootstrap_servers
    )

    producer = KafkaDataclassProducer(kafka_config=kafka_config)

    if args.source == "microphone":
        source_audio = MicrophoneAudioSourceTimed(
            block_duration=args.block_duration, device=args.device
        )
    elif args.source == "file":
        if not args.file_path:
            raise ValueError("An audio file must be specified with --file-path.")
        source_audio = FileAudioSourceTimed(
            file=args.file_path, sample_rate=16000, block_duration=args.block_duration
        )

    logger = logging.getLogger("producer-audio-debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler("producer-audio-debug.log"))

    main(
        producer,
        source_audio=source_audio,
        observer=DebugLogger(logger=logger, skip_keys=["audio_data"]),
    )
