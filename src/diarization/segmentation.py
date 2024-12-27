from typing import Union, List
from uuid import uuid4

import numpy as np
import torch

from diart.models import SegmentationModel
from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata
from src.schema.segmentation import SpeakerSegment


def extract_audio_segment(
    waveform: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    start: int,
    end: int,
    min_duration: float = 0.5,
):
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)

    segment_waveform = waveform[:, start_sample:end_sample]

    # Vérifier si le segment est trop court
    min_samples = int(min_duration * sample_rate)
    if segment_waveform.shape[-1] < min_samples:
        # Ajouter du padding pour atteindre la taille minimale
        padding = min_samples - segment_waveform.shape[-1]
        segment_waveform = torch.nn.functional.pad(
            segment_waveform, (0, padding), "constant", 0
        )

    return segment_waveform


class AudioSegmentation:
    def __init__(
        self,
        segmentation_model: SegmentationModel = SegmentationModel.from_pretrained(
            "pyannote/segmentation", use_hf_token=False
        ),
        target_sample_rate: int = 16_000,
    ):
        self.segmentation_model = segmentation_model
        self.target_sample_rate = target_sample_rate

    @staticmethod
    def get_segments(
        segmentation_result, frame_duration, source_id: str, threshold=0.5
    ) -> List[SpeakerSegment]:
        """
        Convertit la sortie brute de la segmentation en segments temporels.

        Parameters:
        -----------
        segmentation_result : torch.Tensor
            Résultats de segmentation de forme (frames, speakers).
        frame_duration : float
            Durée de chaque frame en secondes.
        threshold : float
            Seuil pour considérer un locuteur comme actif.

        Returns:
        --------
        List[Dict[str, Union[str, float]]]:
            Liste des segments avec début, fin et locuteur.
        """
        segments = []
        num_frames, num_speakers = segmentation_result.shape
        time_stamps = np.arange(num_frames) * frame_duration

        for speaker in range(num_speakers):
            active = segmentation_result[:, speaker] > threshold
            current_segment = None

            for i, is_active in enumerate(active):
                timestamp = time_stamps[i]

                if is_active:
                    if current_segment is None:
                        current_segment = {
                            "source_id": source_id,
                            "start": timestamp,
                            "speaker": f"Speaker {speaker}",
                            "_id": str(uuid4()),
                        }
                    current_segment["end"] = timestamp
                elif current_segment is not None:
                    speaker_segment = SpeakerSegment(**current_segment)
                    segments.append(speaker_segment)
                    current_segment = None

            # Append the last segment if it ends with activity
            if current_segment is not None:
                speaker_segment = SpeakerSegment(**current_segment)
                segments.append(speaker_segment)

        return segments

    def __call__(
        self, audio_metadata: Union[AudioFileMetadata, AudioMicrophoneMetadata]
    ) -> List[SpeakerSegment]:

        segmentation_result = self.segmentation_model(audio_metadata.audio_data)
        audio_duration = audio_metadata.audio_data.shape[-1] / self.target_sample_rate
        frame_duration = (
            audio_duration / segmentation_result.shape[1]
        )  # frames par seconde
        segmentation_result_np = (
            segmentation_result.squeeze(0).detach().numpy()
        )  # (frames, speakers)
        segments = AudioSegmentation.get_segments(
            segmentation_result_np, frame_duration, source_id=audio_metadata._id
        )

        extracted_audio_segments: List[SpeakerSegment] = []
        for segment in segments:
            start, end = segment.start, segment.end
            segment_waveform = extract_audio_segment(
                audio_metadata.audio_data.squeeze(0),
                self.target_sample_rate,
                start,
                end,
            )
            segment.audio_data = segment_waveform
            extracted_audio_segments.append(segment)

        segments_sorted: List[SpeakerSegment] = sorted(
            extracted_audio_segments, key=lambda x: x.start
        )
        return segments_sorted
