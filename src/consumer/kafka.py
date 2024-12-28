from typing import Iterable, Dict, Any, Union, Type, List
import json

from kafka import KafkaConsumer

from src.consumer.base import BaseConsumer
from src.schema.config.consumer import KafkaConsumerConfig

from src.schema.segmentation import SpeakerEmbedding, SpeakerSegment
from src.schema.transcription import Transcription
from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata


class KafkaBaseConsumer(BaseConsumer):

    def __init__(self, kafka_config: KafkaConsumerConfig):
        super().__init__()
        self.enable_auto_commit = kafka_config.enable_auto_commit
        self.consumer = KafkaConsumer(
            kafka_config.topic,
            bootstrap_servers=kafka_config.bootstrap_servers,
            auto_offset_reset=kafka_config.auto_offset_reset,
            enable_auto_commit=self.enable_auto_commit,
            group_id=kafka_config.groud_id,
            fetch_max_bytes=kafka_config.fetch_max_bytes,
        )
        self.topic = kafka_config.topic

    def consume(self):
        return self.read()

    def read(self) -> Iterable[Dict[str, Any]]:
        msg = self.consumer.poll(1.0, max_records=1)
        if msg is None or msg == {}:
            pass
        try:
            for k in msg:
                for record in msg[k]:
                    value = json.loads(record.value.decode("utf-8"))
                    yield value

        except Exception as e:
            raise Exception(str(e))

    def commit(self):
        if not self.enable_auto_commit:
            self.consumer.commit()

    def close(self):
        self.consumer.close()


class KafkaCastConsumer(KafkaBaseConsumer):
    def __init__(
        self,
        kafka_config: KafkaConsumerConfig,
        dataclass: Union[
            Type[SpeakerEmbedding],
            Type[Transcription],
            Type[AudioFileMetadata],
            Type[AudioMicrophoneMetadata],
        ],
    ):
        super().__init__(kafka_config)

        self.dataclass = dataclass

    def read(
        self,
    ) -> Iterable[
        Union[
            List[SpeakerEmbedding],
            List[SpeakerSegment],
            Transcription,
            AudioFileMetadata,
            AudioMicrophoneMetadata,
        ]
    ]:
        reader = super().read()
        if reader is None:
            return []
        for item in reader:
            if isinstance(item, list):
                yield [self.dataclass.from_dict(it) for it in item]
            for it in item:
                yield self.dataclass.from_dict(it)

    def consume(self, value=None) -> Union[
        Type[SpeakerEmbedding],
        Type[Transcription],
        Type[AudioFileMetadata],
        Type[AudioMicrophoneMetadata],
    ]:
        return self.read()
