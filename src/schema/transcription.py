from dataclasses import dataclass
from typing import Any, Optional, Union, Text, Dict
from datetime import datetime
from uuid import uuid4


@dataclass
class Transcription:

    source: str
    audio_segment_index: Union[int, str]
    text: Text
    language: Text
    start_time: Optional[Union[int, datetime]]
    end_time: Optional[Union[int, datetime]]
    extras: Optional[Dict[str, Any]]
    _id: Optional[str] = str(uuid4())
