from typing import Tuple
from pyannote.core.feature import SlidingWindowFeature
from .base import BaseProducer


class DummySegmentationProducer(BaseProducer):
    def __init__(self):
        super().__init__()

    def produce(
        self, value: Tuple[SlidingWindowFeature, SlidingWindowFeature] = None
    ) -> Tuple[SlidingWindowFeature, SlidingWindowFeature]:
        return value
