import json
from dataclasses import asdict

from kafka import KafkaConsumer
from diart.sources import AudioSource


from src.schema.config.consumer import KafkaConsumerConfig
from src.schema.audio import AudioFileMetadata, AudioMicrophoneMetadata
import numpy as np


class KafkaAudioSource(AudioSource):

    def __init__(
        self, kafka_config: KafkaConsumerConfig, sample_rate: int, chunk_size: int
    ):
        super().__init__(uri=kafka_config.topic, sample_rate=sample_rate)
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
        self.chunk_size = chunk_size
        self._stop = False

    def read(self):
        while not self._stop:
            msg = self.consumer.poll(1.0, max_records=1)
            if msg is None or msg == {}:
                continue

            try:
                for k in msg:
                    for record in msg[k]:
                        value = json.loads(record.value.decode("utf-8"))
                        value["audio_data"] = np.asarray(value["audio_data"])
                        data: AudioMicrophoneMetadata = AudioMicrophoneMetadata(**value)
                        msg_metadata = {
                            "topic": record.topic,
                            "partition": record.partition,
                            "offset": record.offset,
                            "timestamp": record.timestamp,
                            "key": record.key.decode("utf-8") if record.key else None,
                        }
                        self.stream.on_next(data)

            except Exception as e:
                print(e)
                self.stream.on_error(e)
                break
        self.stream.on_completed()

    def commit(self):
        if not self.enable_auto_commit:
            self.consumer.commit()

    def close(self):
        self.consumer.close()

    def stop(self):
        self._stop = True
        self.close()
