from dataclasses import dataclass
from typing import Optional, Union
import torch
import numpy as np
from datetime import datetime
from uuid import uuid4


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


@dataclass
class AudioMicrophoneMetadata(AudioFileMetadata): ...
