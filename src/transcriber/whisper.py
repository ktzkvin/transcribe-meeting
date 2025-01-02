import os
import sys
import requests
import whisper
import numpy as np
from dataclasses import asdict
from contextlib import contextmanager
from src.schema.audio import AudioFileMetadata
from src.schema.transcription import Transcription


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class WhisperTranscriber:
    def __init__(self, model="small", device=None):
        self.model = whisper.load_model(model, device=device, in_memory=True)

    def transcribe(self, waveform: AudioFileMetadata) -> Transcription:
        audio = waveform.audio_data.astype("float32").reshape(-1)
        audio = whisper.pad_or_trim(audio)
        with suppress_stdout():
            transcription = self.model.transcribe(
                audio, verbose=False, word_timestamps=True
            )

        return Transcription(
            _id=waveform._id,
            start_time=waveform.start_time,
            end_time=waveform.end_time,
            source=waveform.source,
            audio_segment_index=waveform.audio_segment_index,
            text=transcription.get("text"),
            language=transcription.get("language"),
            extras=transcription,
        )

    def __call__(self, waveform: AudioFileMetadata) -> Transcription:

        return self.transcribe(waveform)


class WhisperApiTranscriber:
    def __init__(
        self,
        api_url="http://localhost:8000/v1/audio/transcriptions/metadata",
    ):
        """
        Initialize the transcriber.

        :param api_url: The URL of the REST API for transcription.
        """
        self.api_url = api_url

    def transcribe(self, waveform: AudioFileMetadata) -> Transcription:
        """
        Perform transcription by calling the REST API.

        :param waveform: The audio metadata object containing audio data and other information.
        :return: A Transcription object with the transcription result.
        """

        headers = {
            "Authorization": "Bearer YOUR_ACCESS_TOKEN"  # Replace with actual token if required
        }

        # Call the API (sending the audio file and form data)
        if len(waveform.audio_data.shape) > 1:
            waveform.audio_data = np.mean(waveform.audio_data, axis=1)

        waveform.audio_data = waveform.audio_data.tolist()
        audio_metadata = asdict(waveform)
        response = requests.post(self.api_url, json=audio_metadata, headers=headers)

        if response.status_code == 200:
            # Assuming the response contains the transcription text and metadata
            transcription_data = response.json()

            # Construct the Transcription object
            transcription = Transcription(
                _id=waveform._id,
                start_time=waveform.start_time,
                end_time=waveform.end_time,
                source=waveform.source,
                audio_segment_index=waveform.audio_segment_index,
                text=transcription_data.get("transcription", ""),
                language=transcription_data.get("language", "en"),
                extras=transcription_data,
            )

            return transcription
        else:
            raise Exception(f"Error from API: {response.status_code}, {response.text}")

    def __call__(self, waveform: AudioFileMetadata) -> Transcription:
        """
        Allow calling the object directly to transcribe the waveform.

        :param waveform: The audio metadata object containing audio data and other information.
        :return: A Transcription object with the transcription result.
        """
        return self.transcribe(waveform)
