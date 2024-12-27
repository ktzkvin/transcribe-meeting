from typing import Optional
from dataclasses import dataclass


@dataclass
class KafkaProducerConfig:
    topic: str = "audio-files"
    bootstrap_servers: str = "0.0.0.0:9093"
    fetch_max_bytes: int = 16_000_000
