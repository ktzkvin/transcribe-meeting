from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum
import json

import torch


class SegmentationModel(str, Enum):
    PYANNOTE_SEGMENTATION = "pyannote/segmentation-3.0"
    ANOTHER_SEGMENTATION = "pyannote/segmentation"


class EmbeddingModel(str, Enum):
    WESPEAKER_VOCELEB = "pyannote/wespeaker-voxceleb-resnet34-LM"


class ConfigError(Exception):
    pass


@dataclass
class AudioConfig:

    segmentation_model: SegmentationModel
    embedding_model: EmbeddingModel

    sample_rate: int = 16000
    mic: Optional[str] = None
    audio_path: Optional[str] = None

    duration: float = 5
    step: float = 0.5
    latency: Optional[Literal["max", "min"]] = None

    tau_active: float = 0.6
    rho_update: float = 0.3
    delta_new: float = 1
    gamma: float = 3
    beta: float = 10
    max_speakers: int = 20

    normalize_embedding_weights: bool = False
    device: Optional[Literal["cuda", "cpu"]] = "cpu"


def load_config_from_json(json_file_path: str) -> AudioConfig:
    with open(json_file_path, "r") as f:
        config_data = json.load(f)

    config_data["segmentation_model"] = SegmentationModel(
        config_data["segmentation_model"]
    )
    config_data["embedding_model"] = EmbeddingModel(config_data["embedding_model"])

    return AudioConfig(**config_data)
