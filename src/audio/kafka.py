import json
import io
import base64
from typing import Tuple, Any
from dataclasses import asdict
import soundfile as sf
import numpy as np

from src.schema.kafka.audio import AudioMetadata, AudioMetadataTyped


def decode_and_read_audio_bytes(audio_data: str) -> Tuple[np.ndarray, Any]:
    audio_str = (
        audio_data.split(",")[2]
        if "data:video" in audio_data
        else audio_data.split(",")[1]
    )
    audio_bytes = base64.b64decode(audio_str)
    waveform, samplerate = sf.read(file=io.BytesIO(audio_bytes), dtype="float32")

    return waveform, samplerate


def audio_mcr_deserializer(audio_metadata: AudioMetadata) -> AudioMetadataTyped:
    audio_np, sample_rate = decode_and_read_audio_bytes(audio_metadata.audio_data)

    audio_metadata_dict = asdict(audio_metadata)
    audio_metadata_dict["audio_data"] = audio_np
    audio_metadata_dict["sample_rate"] = sample_rate

    return AudioMetadataTyped(**audio_metadata_dict)


def process_message_audio_kafka(message: str) -> AudioMetadataTyped:
    audio_metadata = AudioMetadata(**json.loads(message.decode("utf-8")))
    return audio_mcr_deserializer(audio_metadata=audio_metadata)
