from typing import Tuple
from pyannote.core.feature import SlidingWindowFeature
from .base import BaseConsumer




class DummySegmentationConsumer(BaseConsumer):
    def __init__(self):
        super().__init__()
        
    def consume(self, value : Tuple[SlidingWindowFeature, SlidingWindowFeature] = None) -> Tuple[SlidingWindowFeature, SlidingWindowFeature]:
        return value
        