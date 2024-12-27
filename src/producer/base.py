from typing import Any, Union
from abc import ABC, abstractmethod

import json
from uuid import uuid4
from dataclasses import is_dataclass, asdict
from datetime import datetime

from rx.core import Observer
import torch
import numpy as np


from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata


class BaseProducer(ABC):

    @abstractmethod
    def producer(self, value: Any) -> Any: ...


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, uuid4):
            return str(obj)
        return super().default(obj)


class DummyDataclassProducer(Observer, BaseProducer):
    def __init__(self, printing: bool = True):
        super().__init__()
        self.printing = printing

    def producer(self, value: Union[AudioFileMetadata, AudioMicrophoneMetadata]) -> str:
        result = json.dumps(value, cls=DataclassJSONEncoder)
        if self.printing:
            print(result)
        return result

    def on_next(self, value: Union[AudioFileMetadata, AudioMicrophoneMetadata]) -> str:

        result = self.producer(value)
        return result

    def on_error(self, error):
        return super().on_error(error)

    def on_completed(self):
        return super().on_completed()
