import pytest
from unittest.mock import patch
from src.schema.audio import AudioFileMetadata
from src.schema.transcription import Transcription
from src.transcriber.whisper import WhisperTranscriber
import numpy as np
import whisper


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


def test_transcribe(audio_metadata):
    with patch("whisper.load_model") as mock_load_model:
        # Mock le modèle retourné par whisper.load_model
        mock_model = mock_load_model.return_value
        # Mock la méthode transcribe du modèle
        mock_model.transcribe.return_value = {
            "text": "This is a mock transcription.",
            "language": "en",
            "word_timestamps": [],
        }

        # Instanciation du transcripteur
        transcriber = WhisperTranscriber(model="small", device="cpu")

        # Appel de la méthode transcribe
        transcription = transcriber.transcribe(audio_metadata)

        # Vérifications
        assert isinstance(
            transcription, Transcription
        ), "The transcription should be an instance of Transcription"

        assert (
            transcription.text == "This is a mock transcription."
        ), "Transcription text is incorrect"
        assert transcription.language == "en", "Language is incorrect"
        assert (
            transcription._id == audio_metadata._id
        ), "The transcription ID should match the audio metadata ID"
        assert (
            transcription.start_time == audio_metadata.start_time
        ), "Start time does not match"
        assert (
            transcription.end_time == audio_metadata.end_time
        ), "End time does not match"
        assert transcription.source == audio_metadata.source, "Source does not match"
        assert (
            transcription.audio_segment_index == audio_metadata.audio_segment_index
        ), "Audio segment index does not match"
        assert transcription.extras == {
            "text": "This is a mock transcription.",
            "language": "en",
            "word_timestamps": [],
        }, "The transcription extras are incorrect"


# Test de la méthode __call__ de WhisperTranscriber (qui appelle transcribe)
def test_whisper_transcriber_call(audio_metadata):
    with patch("whisper.load_model") as mock_load_model:
        # Mock le modèle retourné par whisper.load_model
        mock_model = mock_load_model.return_value
        # Mock la méthode transcribe du modèle
        mock_model.transcribe.return_value = {
            "text": "This is a mock transcription.",
            "language": "en",
            "word_timestamps": [],
        }

        # Instanciation du transcripteur
        transcriber = WhisperTranscriber(model="small", device="cpu")

        # Appel de la méthode via __call__
        transcription = transcriber(audio_metadata)

        # Vérification du type de la transcription
        assert isinstance(
            transcription, Transcription
        ), "The transcription should be an instance of Transcription"
        assert (
            transcription.text == "This is a mock transcription."
        ), "Transcription text is incorrect"
        assert transcription.language == "en", "Language is incorrect"


if __name__ == "__main__":
    pytest.main()
