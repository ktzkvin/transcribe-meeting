import argparse
import json
import logging
from kafka import KafkaConsumer
from src.schema.config.consumer import KafkaConsumerConfig
from src.observers.logger import DebugLogger


def consume_messages(
    topic, bootstrap_servers, group_id, fetch_max_bytes, log_file, write_data
):
    """
    Consumes messages from a Kafka topic and logs them using DebugLogger.

    Args:
        topic (str): Kafka topic to consume from.
        bootstrap_servers (str): Kafka bootstrap servers.
        group_id (str): Kafka consumer group ID.
        fetch_max_bytes (int): Maximum fetch size for Kafka messages.
        log_file (str): File to save logs.
        write_data (bool): Whether to log message data.
    """
    # Set up Kafka Consumer
    kafka_config = KafkaConsumerConfig(
        topic=topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        fetch_max_bytes=fetch_max_bytes,
        groud_id=group_id,
        enable_auto_commit=True,
    )

    consumer = KafkaConsumer(
        kafka_config.topic,
        bootstrap_servers=kafka_config.bootstrap_servers,
        auto_offset_reset=kafka_config.auto_offset_reset,
        enable_auto_commit=kafka_config.enable_auto_commit,
        group_id=kafka_config.groud_id,
        fetch_max_bytes=kafka_config.fetch_max_bytes,
    )

    # Set up logger
    logger = logging.getLogger("kafka-consumer")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(log_file))

    d_logger = DebugLogger(logger=logger, write_data=write_data)

    print(f"Listening for messages on topic '{topic}'...")
    for message in consumer:
        try:
            decoded_message = json.loads(message.value.decode("utf-8"))
            d_logger.on_next(decoded_message)
        except Exception as e:
            print(f"Error processing message: {e}")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Kafka Consumer with Debug Logging")

    parser.add_argument(
        "--topic", type=str, required=True, help="Kafka topic to consume from."
    )
    parser.add_argument(
        "--bootstrap-servers", type=str, required=True, help="Kafka bootstrap servers."
    )
    parser.add_argument(
        "--group-id", type=str, default="group-debug", help="Kafka consumer group ID."
    )
    parser.add_argument(
        "--fetch-max-bytes",
        type=int,
        default=16_000_000,
        help="Maximum fetch size for Kafka messages.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="debugging.log",
        help="Log file to save debug output.",
    )
    parser.add_argument(
        "--write-data", action="store_true", help="Whether to log message data."
    )

    args = parser.parse_args()

    # Call the consume_messages function with parsed arguments
    consume_messages(
        topic=args.topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.group_id,
        fetch_max_bytes=args.fetch_max_bytes,
        log_file=args.log_file,
        write_data=args.write_data,
    )
