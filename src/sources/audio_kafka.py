from dataclasses import asdict

from kafka import KafkaConsumer
from diart.sources import AudioSource

from src.audio.kafka import process_message_audio_kafka
from src.schema.config.audio import KafkaAudioConfig


class KafkaAudioSource(AudioSource):

    def __init__(
        self, kafka_config: KafkaAudioConfig, sample_rate: int, chunk_size: int
    ):
        super().__init__(sample_rate=sample_rate)
        self.consumer = KafkaConsumer(**asdict(kafka_config))
        self.topic = kafka_config.topic
        self.chunk_size = chunk_size
        self._stop = False

    def read(self):
        while not self._stop:
            msg = self.consumer.poll(1.0)  # Attendre un message pendant 1 seconde
            if msg is None:
                continue
            if msg.error():
                print(f"Erreur Kafka : {msg.error()}")
                continue

            try:
                data = process_message_audio_kafka(msg.value())
                audio_chunk = data.audio_data

                yield audio_chunk

            except Exception as e:
                print(f"{e}")

    def stop(self):
        self._stop = True
        self.consumer.close()
