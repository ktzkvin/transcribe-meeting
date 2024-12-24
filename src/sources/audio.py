from datetime import datetime, timedelta
import numpy as np
import torch
from einops import rearrange
from uuid import uuid4


from diart.sources import MicrophoneAudioSource, FileAudioSource

from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata


class FileAudioSourceTimed(FileAudioSource):
    def __init__(self, file, sample_rate, padding=(0, 0), block_duration=0.5):
        super().__init__(file, sample_rate, padding, block_duration)

    def read(self):
        """Send each chunk of samples through the stream"""
        # [WARNING] : this code are from diart package make if you get any error refer to the repo
        waveform = self.loader.load(self.file)

        # Add zero padding at the beginning if required
        if self.padding_start > 0:
            num_pad_samples = int(
                np.rint(self.padding_start * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([zero_padding, waveform], dim=1)

        # Add zero padding at the end if required
        if self.padding_end > 0:
            num_pad_samples = int(np.rint(self.padding_end * self.sample_rate))
            zero_padding = torch.zeros(waveform.shape[0], num_pad_samples)
            waveform = torch.cat([waveform, zero_padding], dim=1)

        # Split into blocks
        _, num_samples = waveform.shape
        chunks = rearrange(
            waveform.unfold(1, self.block_size, self.block_size),
            "channel chunk sample -> chunk channel sample",
        ).numpy()

        # Add last incomplete chunk with padding
        if num_samples % self.block_size != 0:
            last_chunk = (
                waveform[:, chunks.shape[0] *
                         self.block_size:].unsqueeze(0).numpy()
            )
            diff_samples = self.block_size - last_chunk.shape[-1]
            last_chunk = np.concatenate(
                [last_chunk, np.zeros((1, 1, diff_samples))], axis=-1
            )
            chunks = np.vstack([chunks, last_chunk])

        # Stream blocks
        for i, waveform in enumerate(chunks):

            try:
                if self.is_closed:
                    break
                audio_metadata = AudioFileMetadata(
                    audio_data=waveform,
                    start_time=0,
                    end_time=self.duration,
                    sample_rate=self.sample_rate,
                    source=self.file,
                    audio_segment_index=i,
                    _id=str(uuid4())

                )
                self.stream.on_next(audio_metadata)
            except BaseException as e:
                self.stream.on_error(e)
                break
        self.stream.on_completed()
        self.close()

    def close(self):
        self.is_closed = True


class MicrophoneAudioSourceTimed(MicrophoneAudioSource):
    """Audio source tied to a local microphone.

    Parameters
    ----------
    block_duration: int
        Duration of each emitted chunk in seconds.
        Defaults to 0.5 seconds.
    device: int | str | (int, str) | None
        Device identifier compatible for the sounddevice stream.
        If None, use the default device.
        Defaults to None.
    """

    def __init__(self, block_duration=0.5, device=None):
        super().__init__(block_duration, device)
        self.block_duration = block_duration

    def _read_callback(self, samples, *args):
        now = datetime.now()

        audio_metadata = AudioMicrophoneMetadata(
            source=self.uri,
            audio_data=samples[:, [0]].T,
            start_time=now - timedelta(seconds=self.block_duration),
            end_time=now,
            audio_segment_index=0,
            sample_rate=self.sample_rate,
            _id=str(uuid4())
        )
        self._queue.put_nowait(audio_metadata)
