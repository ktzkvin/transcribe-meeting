import argparse
import logging
import traceback
import rx.operators as ops

from src.sources.audio_kafka import KafkaAudioSource
from src.schema.config.consumer import KafkaConsumerConfig

from src.observers.logger import DebugAudioMetadataLogger
from rx.core import Observer


def main(source_audio: KafkaAudioSource, observer: Observer):
    source_audio.stream.pipe(ops.do(observer)).subscribe(
        on_error=lambda _: traceback.print_exc()
    )
    source_audio.read()


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Kafka Audio Consumer with Debugging")
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
    parser.add_argument(
        "--log-file",
        type=str,
        default="audio-debug.log",
        help="Log file for saving debug output (default: 'audio-debug.log')",
    )
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="Whether to log audio data (default: False)",
    )

    args = parser.parse_args()

    # Set up the Kafka audio source
    source_audio = KafkaAudioSource(
        kafka_config=KafkaConsumerConfig(
            topic=args.topic, bootstrap_servers=args.bootstrap_servers
        ),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )

    # Set up the logger
    logger = logging.getLogger("debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(args.log_file))

    # Create the observer with the provided logging settings
    observer = DebugAudioMetadataLogger(logger=logger, write_data=args.write_data)

    # Run the main function
    main(source_audio=source_audio, observer=observer)