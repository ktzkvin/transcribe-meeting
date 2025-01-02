from typing import Union, Optional
from datetime import datetime
from dataclasses import dataclass
from uuid import uuid4
import numpy as np
import torch


@dataclass
class Segment:
    _id: str
    source_id: str
    start: Union[int, datetime]
    end: Union[int, datetime]

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        if "_id" not in data:
            data["_id"] = str(uuid4())
        return Segment(**data)


@dataclass
class SpeakerSegment(Segment):
    speaker: str
    audio_data: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerSegment":
        audio_data = data.get("audio_data")
        if isinstance(audio_data, list):
            audio_data = torch.FloatTensor(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)

        data["audio_data"] = audio_data

        if "_id" not in data:
            data["_id"] = str(uuid4())
        return SpeakerSegment(**data)


@dataclass
class SpeakerEmbedding(SpeakerSegment):
    embedding: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        audio_data = data.get("audio_data")
        if isinstance(audio_data, list):
            audio_data = torch.FloatTensor(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)

        data["audio_data"] = audio_data

        embedding = data.get("embedding")
        if isinstance(embedding, list):
            embedding = torch.FloatTensor(embedding)
        elif isinstance(audio_data, np.ndarray):
            embedding = torch.from_numpy(embedding)

        data["embedding"] = embedding

        if "_id" not in data:
            data["_id"] = str(uuid4())

        return SpeakerEmbedding(**data)
