import argparse
import logging
import os
import traceback
from dotenv import load_dotenv
from huggingface_hub import login
from rx.core import Observer
import rx.operators as ops


from src.sources.audio_kafka import KafkaAudioSource
from src.schema.config.consumer import KafkaConsumerConfig
from src.schema.config.producer import KafkaProducerConfig
from src.observers.logger import DebugAudioMetadataLogger, DebugLogger
from src.diarization.segmentation import AudioSegmentation
from src.producer.kafka import KafkaDataclassProducer

# Load environment variables
load_dotenv()
login(token=os.environ["HF_TOKEN"])


def main(
    audio: KafkaAudioSource,
    audio_segmentation: AudioSegmentation,
    observer_audio: Observer,
    segmentation_observer: Observer,
    producer_segmentation: KafkaDataclassProducer,
):
    """Main function to process Kafka audio stream."""
    audio.stream.pipe(
        ops.do(observer_audio),
        ops.map(audio_segmentation),
        ops.do(segmentation_observer),
        ops.do(producer_segmentation),
        ops.map(lambda _: audio.commit()),
    ).subscribe(on_error=lambda _: traceback.print_exc())

    print("*" * 79)
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
        default="consumer-segmentation-debug.log",
        help="Log file for saving debug output (default: 'consumer-segmentation-debug.log')",
    )
    parser.add_argument(
        "--write-data",
        action="store_true",
        help="Whether to log audio data (default: False)",
    )

    # Device settings
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
        kafka_config=KafkaConsumerConfig(
            topic=args.topic,
            bootstrap_servers=args.bootstrap_servers,
            groud_id="segmentation-consumer-audio",
            enable_auto_commit=False,
        ),
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
    )

    # Set up loggers
    logger = logging.getLogger("consumer-audio-debug")
    logger_segmentation = logging.getLogger("consumer-segmentation-debug")
    logger.setLevel(logging.DEBUG)
    logger_segmentation.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.log_file))
    logger_segmentation.addHandler(
        logging.FileHandler("consumer-segmentation-debug.log")
    )

    # Create observers
    observer = DebugAudioMetadataLogger(logger=logger, write_data=args.write_data)
    segmentation_observer = DebugLogger(
        logger=logger_segmentation, skip_keys=["audio_data"]
    )
    producer = KafkaDataclassProducer(
        kafka_config=KafkaProducerConfig(
            topic="segmentation-topic",
            bootstrap_servers=args.bootstrap_servers,
            fetch_max_bytes=16_000_000,
        )
    )

    # Run the main function
    main(
        audio=source_audio,
        audio_segmentation=AudioSegmentation(),
        observer_audio=observer,
        segmentation_observer=segmentation_observer,
        producer_segmentation=producer,
    )
