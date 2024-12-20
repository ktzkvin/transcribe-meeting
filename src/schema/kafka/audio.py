from dataclasses import dataclass
from typing import Any, Optional, Union
import torch
import numpy as np


# https://github.com/IA-Generative/mcr-core/blob/43d7f8f1afe2611e29ae8b1fb02ecba0f5f249cf/mcr_meeting/app/services/comu_audio_service.py#L148
@dataclass
class AudioMetadata:
    audio_data: Any  # audio_data (str): Audio data (base64-encoded bytes).
    start_time: Optional[Any]
    end_time: Optional[Any]
    speakers: bool
    meeting_id: str
    audio_segment_index: int
    sample_rate: Optional[Any]


@dataclass
class AudioMetadataTyped:
    audio_data: Union[torch.Tensor, np.ndarray]
    start_time: Optional[Any]
    end_time: Optional[Any]
    speakers: bool
    meeting_id: str
    audio_segment_index: int
    sample_rate: Optional[Any]
