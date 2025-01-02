from typing import Optional, Union, List
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class KafkaConsumerConfig:
    topic: Union[List[str], str] = "audio-files"
    bootstrap_servers: str = "0.0.0.0:9093"
    auto_offset_reset: str = "earliest"
    fetch_max_bytes: int = 16_000_000
    groud_id: Optional[str] = str(uuid4())
    enable_auto_commit: Optional[bool] = True
