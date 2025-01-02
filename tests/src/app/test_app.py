import os
import pytest
from fastapi.testclient import TestClient
import soundfile as sf
import numpy as np
from src.app.app import app

client = TestClient(app)


def test_transcribe_file():
    """Test pour la route /v1/audio/transcriptions avec un fichier audio."""

    file_path = "data/audios/audio_DER.wav"
    if os.path.exists(file_path):

        with open(file_path, "rb") as audio_file:
            files = {"file": ("audio_DER.wav", audio_file, "audio/wav")}
            form_data = {
                "model": "tiny",
                "language": "en",
                "prompt": "",
                "response_format": "json",
                "temperature": "0",
                "timestamp_granularities": "['word']",
            }
            headers = {"Authorization": "Bearer dummy_token"}

            response = client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=form_data,
                headers=headers,
            )

        assert (
            response.status_code == 200
        ), f"Unexpected status code: {response.status_code}"
        assert "metadata" in response.json(), "metadata information missing in response"


def test_transcribe_metadata():
    """Test pour la route /v1/audio/transcriptions/metadata avec métadonnées audio."""
    audio_metadata = {
        "audio_data": [0.0, 0.1, -0.1, 0.2],
        "start_time": 0.0,
        "end_time": 30.0,
        "source": "example.wav",
        "audio_segment_index": 0,
        "sample_rate": 16000,
    }
    headers = {"Authorization": "Bearer dummy_token"}

    response = client.post(
        "/v1/audio/transcriptions/metadata",
        json=audio_metadata,
        headers=headers,
    )

    assert (
        response.status_code == 200
    ), f"Unexpected status code: {response.status_code}"
    assert "metadata" in response.json(), "metadata information missing in response"


def test_transcribe_metadata_with_audio_content():
    """Test pour la route /v1/audio/transcriptions/metadata avec le contenu réel du fichier audio."""
    # Chemin vers le fichier audio existant
    file_path = "data/audios/audio_DER.wav"

    if os.path.exists(file_path):

        audio, sample_rate = sf.read(file_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio_metadata = {
            "audio_data": audio.tolist(),
            "start_time": 0.0,
            "end_time": len(audio) / sample_rate,
            "source": "audio_DER.wav",
            "audio_segment_index": 0,
            "sample_rate": sample_rate,
        }

        headers = {"Authorization": "Bearer dummy_token"}

        response = client.post(
            "/v1/audio/transcriptions/metadata",
            json=audio_metadata,
            headers=headers,
        )

        assert (
            response.status_code == 200
        ), f"Unexpected status code: {response.status_code}"

        assert "metadata" in response.json(), "metadata information missing in response"


if __name__ == "__main__":
    pytest.main()
