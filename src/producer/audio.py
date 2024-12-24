from typing import Union
import json
from uuid import uuid4
from dataclasses import is_dataclass, asdict
from datetime import datetime

from rx.core import Observer
import torch
import numpy as np


from .base import BaseProducer
from ..schema.audio import AudioFileMetadata, AudioMicrophoneMetadata
from ..schema.config.audio import KafkaAudioConfig


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


class DummyAudioProducer(Observer, BaseProducer):
    def __init__(self, printing: bool = True):
        super().__init__()
        self.printing = printing

    def producer(self, value: Union[AudioFileMetadata, AudioMicrophoneMetadata]) -> str:
        result = json.dumps(value, cls=DataclassJSONEncoder)
        if self.printing:
            print(result)
        return result

    def on_next(
        self, value: Union[AudioFileMetadata, AudioMicrophoneMetadata]
    ) -> str:

        result = self.producer(value)
        return result

    def on_error(self, error):
        return super().on_error(error)

    def on_completed(self):
        return super().on_completed()


class KafkaAudioProducer(DummyAudioProducer):
    def __init__(self, kafka_config: KafkaAudioConfig):
        super().__init__(printing=False)
        try:

            from kafka import KafkaProducer

        except Exception as e:
            raise ImportError(str(e))

        self.__producer: KafkaProducer = KafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,)
        self.__topic = kafka_config.topic
        self.counter = 0

    def producer(self, value):
        message = super().producer(value)
        self.__producer.send(topic=self.__topic, value=message.encode('utf-8'))
        print(f"OK {self.counter}")
        self.counter += 1
        return message

    def on_completed(self):
        self.__producer.close()
        return super().on_completed()
