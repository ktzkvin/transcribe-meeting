from dataclasses import dataclass
from typing import Optional, Union, List
import torch
import numpy as np
from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel


@dataclass
class AudioFileMetadata:
    audio_data: Union[
        torch.Tensor, np.ndarray
    ]  # audio_data (str): Audio data (base64-encoded bytes).
    start_time: Optional[Union[int, datetime]]
    end_time: Optional[Union[int, datetime]]
    source: str
    audio_segment_index: int
    sample_rate: Optional[int]
    _id: Optional[str] = str(uuid4())

    @classmethod
    def from_dict(cls, data: dict) -> "AudioFileMetadata":
        audio_data = data["audio_data"]
        if isinstance(audio_data, list):
            audio_data = np.asarray(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data)

        data["audio_data"] = audio_data
        if "_id" not in data:
            data["_id"] = str(uuid4())

        return AudioFileMetadata(**data)


@dataclass
class AudioMicrophoneMetadata(AudioFileMetadata): ...


class AudioMetadataInput(BaseModel):
    audio_data: List[float]
    start_time: Optional[float]
    end_time: Optional[float]
    source: str
    audio_segment_index: int
    sample_rate: Optional[int]
