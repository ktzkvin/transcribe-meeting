import unittest
from unittest.mock import MagicMock
import torch
from src.schema.segmentation import SpeakerSegment, SpeakerEmbedding
from src.diarization.embedding import AudioEmbedding


class TestAudioEmbedding(unittest.TestCase):

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.embedding_model.return_value = torch.rand(
            1, 512
        )  # Embedding size 512
        self.audio_embedding = AudioEmbedding(embedding_model=self.mock_model)

    def test_call_method(self):
        # Create mock segments
        segments = [
            SpeakerSegment(
                source="test_source",
                start=0.0,
                end=1.0,
                speaker="Speaker 0",
                audio_data=torch.rand(1, 16000),  # 1 second audio at 16kHz
            ),
            SpeakerSegment(
                source="test_source",
                start=1.0,
                end=2.0,
                speaker="Speaker 1",
                audio_data=torch.rand(1, 16000),  # 1 second audio at 16kHz
            ),
        ]

        # Call the audio embedding
        embeddings = self.audio_embedding(segments)

        # Validate the results
        self.assertEqual(len(embeddings), len(segments))
        for embedding, segment in zip(embeddings, segments):
            self.assertIsInstance(embedding, SpeakerEmbedding)
            self.assertEqual(embedding.source, segment.source)
            self.assertEqual(embedding.start, segment.start)
            self.assertEqual(embedding.end, segment.end)
            self.assertEqual(embedding.speaker, segment.speaker)
            self.assertTrue(
                torch.allclose(torch.tensor(embedding.audio_data), segment.audio_data)
            )
            self.assertEqual(
                embedding.embedding.shape, (1, 512)
            )  # Check embedding size


if __name__ == "__main__":
    unittest.main()
