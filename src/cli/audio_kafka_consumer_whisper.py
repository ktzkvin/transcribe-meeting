from typing import Union
import argparse
import logging
import traceback
import rx.operators as ops
from rx.core import Observer

from src.sources.audio_kafka import KafkaAudioSource
from src.schema.config.consumer import KafkaConsumerConfig
from src.schema.config.producer import KafkaProducerConfig

from src.observers.logger import DebugAudioMetadataLogger, DebugLogger
from src.transcriber.whisper import WhisperTranscriber, WhisperApiTranscriber
from src.producer.kafka import KafkaDataclassProducer


def main(
    source_audio: KafkaAudioSource,
    observer_audio: Observer,
    transcriber: Union[WhisperTranscriber, WhisperApiTranscriber],
    producer: KafkaDataclassProducer,
    observer_transcription: Observer,
):
    source_audio.stream.pipe(
        ops.do(observer_audio),
        ops.map(transcriber),
        ops.do(observer_transcription),
        ops.do(producer),
        ops.map(lambda _: source_audio.commit()),
    ).subscribe(on_error=lambda _: traceback.print_exc())
    source_audio.read()


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
        default="consumer-whisper-debug.log",
        help="Log file for saving debug output (default: 'consumer-whisper-debug.log')",
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

    # Optional transcription API URL (for WhisperApiTranscriber)
    parser.add_argument(
        "--url-transcription",
        type=str,
        default=None,
        help="API URL for external transcription service (optional)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up the Kafka audio source
    source_audio = KafkaAudioSource(
        kafka_config=KafkaConsumerConfig(
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            group_id="consumer-audio",
            enable_auto_commit=False,
        ),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )

    producer = KafkaDataclassProducer(
        kafka_config=KafkaProducerConfig(
            topic="whisper",
            bootstrap_servers=args.bootstrap_servers,
        )
    )

    # Set up the logger
    input_logger = logging.getLogger("input-whisper-debug")
    input_logger.setLevel(logging.DEBUG)
    input_logger.addHandler(logging.FileHandler(args.log_file))

    output_logger = logging.getLogger("output-whisper-debug")
    output_logger.setLevel(logging.DEBUG)
    output_logger.addHandler(logging.FileHandler("output-whisper-debug.log"))

    # Create the observer with the provided logging settings
    observer = DebugAudioMetadataLogger(logger=input_logger, write_data=args.write_data)

    # Set up Whisper transcriber
    if args.url_transcription is None:
        transcriber = WhisperTranscriber(model=args.model, device=args.device)
    else:
        transcriber = WhisperApiTranscriber(api_url=args.url_transcription)

    # Run the main function
    main(
        source_audio=source_audio,
        observer_audio=observer,
        transcriber=transcriber,
        producer=producer,
        observer_transcription=DebugLogger(logger=output_logger),
    )
