from dataclasses import dataclass


@dataclass
class KafkaAudioConfig:
    topic: str = "audio-files"
    bootstrap_servers: str = "0.0.0.0:9093"
    auto_offset_reset: str = "earliest"
    fetch_max_bytes: int = 10_485_760
