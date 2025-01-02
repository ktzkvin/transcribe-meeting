from dataclasses import dataclass
from typing import Any, Optional, Union, Text, Dict
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel
from src.schema.app import Metadata


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

    @classmethod
    def from_dict(cls, data: dict) -> "Transcription":
        if "_id" not in data:
            data["_id"] = str(uuid4())

        return Transcription(**data)


class MetadataTranscription(Metadata):
    model_version: str
    inference_time: float
    read_time: float


class TranscriptionOutput(BaseModel):
    source: str
    audio_segment_index: Union[int, str]
    text: Text
    language: Text
    metadata: MetadataTranscription
    start_time: Optional[Union[int, datetime]]
    end_time: Optional[Union[int, datetime]]
    extras: Optional[Dict[str, Any]]
    _id: Optional[str] = str(uuid4())
