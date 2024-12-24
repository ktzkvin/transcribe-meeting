from typing import Tuple, List, Dict, Any

from rx.core import Observer
from pyannote.core.feature import SlidingWindowFeature
from pyannote.core.annotation import Annotation
from datetime import datetime, timedelta

from .base import BaseProducer


class JsonStdoutProducer(Observer, BaseProducer):
    def __init__(self, _id: str, start_time: datetime = datetime.now()):
        super().__init__()
        self.start_time = start_time
        self.id = _id

    def producer(self, value):
        print(value)

    def on_next(
        self, value: Tuple[Annotation, SlidingWindowFeature]
    ) -> List[Dict[str, Any]]:
        logs = []
        annotation, slide = value
        for s, t, l in annotation.itertracks(yield_label=True):
            log = {
                "uri": annotation.uri,
                "start": self.start_time + timedelta(seconds=s.start),
                "duration": s.duration,
                "end": self.start_time + timedelta(seconds=s.end),
                "label": l,
                "slide.dimension": slide.dimension,
                "slide.labels": slide.labels,
                "track": t,
                "id": self.id,
            }
            logs.append(log)

        for log in logs:
            self.producer(log)
        return logs

    def on_error(self, error):
        return super().on_error(error)

    def on_completed(self):
        return super().on_completed()
