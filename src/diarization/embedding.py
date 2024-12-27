from typing import List

from diart.models import EmbeddingModel
from src.schema.segmentation import SpeakerEmbedding, SpeakerSegment


class AudioEmbedding:
    def __init__(
        self,
        embedding_model: EmbeddingModel = EmbeddingModel.from_pretrained(
            "pyannote/embedding", use_hf_token=False
        ),
    ):

        self.embedding_model = embedding_model

    def __call__(self, segments: List[SpeakerSegment]) -> List[SpeakerEmbedding]:
        speakers_embedding: List[SpeakerEmbedding] = []
        for segment in segments:
            audio_sample = segment.audio_data.unsqueeze(0)
            embedding = self.embedding_model.embedding_model(audio_sample)
            speaker_embedding = SpeakerEmbedding(
                _id=segment._id,
                source_id=segment.source_id,
                start=segment.start,
                end=segment.end,
                speaker=segment.speaker,
                audio_data=segment.audio_data,
                embedding=embedding.detach().numpy(),
            )
            speakers_embedding.append(speaker_embedding)

        return speakers_embedding
