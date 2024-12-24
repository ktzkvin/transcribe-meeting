import os
import sys
import whisper
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
