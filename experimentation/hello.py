import os 
from typing import Union, Tuple
from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter, Observer, _extract_prediction
from diart import SpeakerDiarizationConfig

from pyannote.core import Annotation
from huggingface_hub import login


login(token=os.environ['HF_TOKEN'])


class PrintObserver(Observer):
    def __init__(self):
        super().__init__()

    def patch(self):
        """Stitch same-speaker turns that are close to each other"""
        pass

    def on_next(self, value: Union[Tuple, Annotation]):
        prediction = _extract_prediction(value)
        # Write prediction in RTTM format
        print(prediction)

    def on_error(self, error: Exception):
        self.patch()

    def on_completed(self):
        self.patch()

pipeline = SpeakerDiarization(SpeakerDiarizationConfig(device='cuda'))
mic = MicrophoneAudioSource()
file_audio = FileAudioSource(file="data/audios/airport_audio.wav", sample_rate=16000)
inference = StreamingInference(pipeline, file_audio, do_plot=False)
inference.attach_observers(RTTMWriter(mic.uri, "file.rttm"), PrintObserver())
prediction = inference()