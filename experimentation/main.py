import rx.operators as ops
import diart.operators as dops
from diart.sources import FileAudioSource
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding
from pyannote.core.feature import SlidingWindowFeature

segmentation = SpeakerSegmentation.from_pretrained("pyannote/segmentation-3.0")
embedding = OverlapAwareSpeakerEmbedding.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
mic = FileAudioSource(file="audios/audio.wav", sample_rate=16000)


def apply_segmentation(wav):
    seg : SlidingWindowFeature = segmentation(wav)
    
    print(seg.data.shape)
    return seg
stream = mic.stream.pipe(
    # Reformat stream to 5s duration and 500ms shift
    dops.rearrange_audio_stream(sample_rate=16000),
    ops.map(lambda wav: (wav, apply_segmentation(wav))),
    ops.starmap(embedding)
).subscribe(on_next=lambda emb: print(emb.shape))

mic.read()
