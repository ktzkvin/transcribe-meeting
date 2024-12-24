import unittest
from unittest.mock import patch, MagicMock
import base64
import json
import numpy as np
from io import BytesIO
from dataclasses import asdict
from src.audio.kafka import (
    AudioMetadata,
    AudioMetadataTyped,
    decode_and_read_audio_bytes,
    audio_mcr_deserializer,
    process_message_audio_kafka,
)


class TestAudioProcessing(unittest.TestCase):

    def setUp(self):
        self.sample_audio = np.random.rand(16000).astype("float32")
        self.sample_rate = 16000
        self.buffer = BytesIO()
        import soundfile as sf

        sf.write(self.buffer, self.sample_audio, self.sample_rate, format="WAV")
        self.buffer.seek(0)
        self.audio_bytes = self.buffer.read()
        self.audio_base64 = base64.b64encode(self.audio_bytes).decode("utf-8")
        self.audio_data = f"data:audio/wav;base64,{self.audio_base64}"

        self.metadata = AudioMetadata(
            audio_data=self.audio_data,
            start_time=None,
            end_time=None,
            speakers=True,
            meeting_id="12345",
            audio_segment_index=1,
            sample_rate=None,
        )

    @patch("src.audio.kafka.sf.read")
    def test_decode_and_read_audio_bytes(self, mock_sf_read):
        mock_sf_read.return_value = (self.sample_audio, self.sample_rate)
        waveform, samplerate = decode_and_read_audio_bytes(self.audio_data)

        self.assertTrue(np.array_equal(waveform, self.sample_audio))
        self.assertEqual(samplerate, self.sample_rate)
        mock_sf_read.assert_called_once()

    @patch("src.audio.kafka.decode_and_read_audio_bytes")
    def test_audio_mcr_deserializer(self, mock_decode):
        mock_decode.return_value = (self.sample_audio, self.sample_rate)

        audio_typed = audio_mcr_deserializer(self.metadata)

        self.assertTrue(np.array_equal(audio_typed.audio_data, self.sample_audio))
        self.assertEqual(audio_typed.sample_rate, self.sample_rate)
        self.assertEqual(audio_typed.meeting_id, self.metadata.meeting_id)
        self.assertEqual(
            audio_typed.audio_segment_index, self.metadata.audio_segment_index
        )

    @patch("src.audio.kafka.audio_mcr_deserializer")
    def test_process_message_audio_kafka(self, mock_deserializer):
        mock_typed_metadata = AudioMetadataTyped(
            audio_data=self.sample_audio,
            start_time=None,
            end_time=None,
            speakers=True,
            meeting_id="12345",
            audio_segment_index=1,
            sample_rate=self.sample_rate,
        )
        mock_deserializer.return_value = mock_typed_metadata

        message = json.dumps(asdict(self.metadata)).encode("utf-8")
        audio_typed = process_message_audio_kafka(message)

        self.assertEqual(audio_typed.meeting_id, mock_typed_metadata.meeting_id)
        self.assertEqual(
            audio_typed.audio_segment_index, mock_typed_metadata.audio_segment_index
        )
        self.assertTrue(np.array_equal(audio_typed.audio_data, self.sample_audio))
        self.assertEqual(audio_typed.sample_rate, self.sample_rate)
        mock_deserializer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
