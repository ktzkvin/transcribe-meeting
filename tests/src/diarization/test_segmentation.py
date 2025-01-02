import unittest
from unittest.mock import MagicMock
import torch
from src.schema.audio import AudioFileMetadata
from src.schema.segmentation import SpeakerSegment
from src.diarization.segmentation import AudioSegmentation, extract_audio_segment


class TestExtractAudioSegment(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 16000

    def test_extract_segment_within_bounds(self):
        waveform = torch.rand(1, self.sample_rate * 10)  # 10 seconds audio
        start, end = 2, 4  # Extract 2 seconds segment
        segment = extract_audio_segment(waveform, self.sample_rate, start, end)
        self.assertEqual(segment.shape[-1], self.sample_rate * (end - start))

    def test_extract_segment_with_padding(self):
        waveform = torch.rand(1, self.sample_rate * 10)  # 10 seconds audio
        start, end = 8, 8.2  # Extract 0.2 seconds segment
        min_duration = 0.5
        segment = extract_audio_segment(
            waveform, self.sample_rate, start, end, min_duration
        )
        self.assertEqual(segment.shape[-1], int(self.sample_rate * min_duration))

    def test_extract_segment_out_of_bounds(self):
        waveform = torch.rand(1, self.sample_rate * 5)  # 5 seconds audio
        start, end = 4, 6  # Request 2 seconds starting at 4s
        segment = extract_audio_segment(waveform, self.sample_rate, start, end)
        self.assertEqual(
            segment.shape[-1], self.sample_rate * 1
        )  # Only 1 second available


class TestAudioSegmentation(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.return_value = torch.rand(
            1, 100, 2
        )  # (batch, frames, speakers)
        self.audio_segmentation = AudioSegmentation(
            segmentation_model=self.mock_model, target_sample_rate=16000
        )

    def test_get_segments(self):
        segmentation_result = torch.tensor(
            [[0.6, 0.2], [0.7, 0.1], [0.4, 0.8], [0.5, 0.6]]
        )
        frame_duration = 0.1
        source = "test_source"
        segments = AudioSegmentation.get_segments(
            segmentation_result, frame_duration, source, threshold=0.5
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].speaker, "Speaker 0")
        self.assertEqual(segments[1].speaker, "Speaker 1")
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[0].end, 0.1)

    def test_call_method(self):
        audio_data = torch.rand(1, 16000 * 5)  # 5 seconds audio
        audio_metadata = AudioFileMetadata(
            audio_data=audio_data.unsqueeze(0),
            source="test_source",
            start_time=0,
            end_time=5,
            audio_segment_index=0,
            sample_rate=16_000,
        )

        # Mock segmentation model output
        self.mock_model.return_value = torch.rand(
            1, 100, 2
        )  # (batch, frames, speakers)
        segments = self.audio_segmentation(audio_metadata)

        # Check segments are returned
        self.assertGreater(len(segments), 0)
        for segment in segments:
            self.assertIsInstance(segment, SpeakerSegment)
            self.assertTrue(hasattr(segment, "audio_data"))


if __name__ == "__main__":
    unittest.main()
