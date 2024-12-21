import unittest
from unittest.mock import patch
from pydub import AudioSegment
from src.consumer.audio import FileAudioConsumer


class TestFileAudioConsumer(unittest.TestCase):
    @patch("pydub.AudioSegment.from_file")
    def test_consume_chunks(self, mock_from_file):
        mock_audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
        mock_from_file.return_value = mock_audio

        consumer = FileAudioConsumer("mock_file.mp3", 2)  # 2-second chunks
        chunks = list(consumer.consume())

        self.assertEqual(len(chunks), 3)  # Expecting 3 chunks (2s, 2s, 1s)
        self.assertEqual(len(chunks[0]), 2000)
        self.assertEqual(len(chunks[1]), 2000)
        self.assertEqual(len(chunks[2]), 1000)


if __name__ == "__main__":
    unittest.main()
