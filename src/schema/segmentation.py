from typing import Union, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass


@dataclass
class Segment:
    source: str
    start: Union[int, datetime]
    end: Union[int, datetime]


@dataclass
class SpeakerSegment(Segment):
    speaker: str
    audio_data: Optional[np.ndarray] = None


@dataclass
class SpeakerEmbedding(SpeakerSegment):
    embedding: Optional[np.ndarray] = None
