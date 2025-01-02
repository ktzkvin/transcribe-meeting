import pytest
from src.schema.audio import AudioFileMetadata
from src.schema.transcription import Transcription
from src.transcriber.whisper import WhisperApiTranscriber
import numpy as np


@pytest.fixture
def audio_metadata():
    return AudioFileMetadata(
        audio_data=np.array([0.5, -0.5, 0.3, 0.2]),
        start_time=0,
        end_time=10,
        source="test_audio.wav",
        audio_segment_index=1,
        sample_rate=16000,
    )


def test_whisper_transcriber_call_real_api(audio_metadata):
    api_url = "http://localhost:8000/v1/audio/transcriptions/metadata"

    transcriber = WhisperApiTranscriber(api_url=api_url)

    transcription = transcriber(audio_metadata)

    assert isinstance(transcription, Transcription), "must be Transcription"


if __name__ == "__main__":
    pytest.main()
