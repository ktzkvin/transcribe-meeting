import unittest
from unittest.mock import MagicMock
import torch
from src.schema.segmentation import SpeakerSegment, SpeakerEmbedding
from src.diarization.embedding import AudioEmbedding


class TestAudioEmbedding(unittest.TestCase):

    def setUp(self):
        # Création du mock de modèle d'embedding
        self.mock_model = MagicMock()
        # On simule que l'appel à `embedding_model()` retourne un tensor de forme (1, 512)
        self.mock_model.embedding_model.return_value = torch.rand(
            1, 512
        )  # Embedding size 512
        self.audio_embedding = AudioEmbedding(embedding_model=self.mock_model)

    def test_call_method(self):
        # Crée des segments simulés
        segments = [
            SpeakerSegment(
                _id="1",
                source_id="test_source",
                start=0.0,
                end=1.0,
                speaker="Speaker 0",
                audio_data=torch.rand(1, 16000),  # Audio de 1 seconde à 16 kHz
            ),
            SpeakerSegment(
                _id="2",
                source_id="test_source",
                start=1.0,
                end=2.0,
                speaker="Speaker 1",
                audio_data=torch.rand(1, 16000),  # Audio de 1 seconde à 16 kHz
            ),
        ]

        # Call the audio embedding
        embeddings = self.audio_embedding(segments)

        # Vérifie les résultats
        self.assertEqual(len(embeddings), len(segments))
        for embedding, segment in zip(embeddings, segments):
            self.assertIsInstance(embedding, SpeakerEmbedding)
            self.assertEqual(embedding.source_id, segment.source_id)
            self.assertEqual(embedding._id, segment._id)
            self.assertEqual(embedding.start, segment.start)
            self.assertEqual(embedding.end, segment.end)
            self.assertEqual(embedding.speaker, segment.speaker)
            self.assertTrue(
                torch.allclose(torch.tensor(embedding.audio_data), segment.audio_data)
            )


if __name__ == "__main__":
    unittest.main()
