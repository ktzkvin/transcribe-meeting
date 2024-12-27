import sys


from src.producer.base import DummyDataclassProducer
from src.schema.config.producer import KafkaTranscriptionConfig


class KafkaDataclassProducer(DummyDataclassProducer):
    def __init__(self, kafka_config: KafkaTranscriptionConfig):
        super().__init__(printing=False)
        try:

            from kafka import KafkaProducer

        except Exception as e:
            raise ImportError(str(e))

        self.__producer: KafkaProducer = KafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            max_request_size=kafka_config.fetch_max_bytes,
        )
        self.__topic = kafka_config.topic
        self.counter = 0

    def producer(self, value):
        message = super().producer(value)
        value = message.encode("utf-8")
        size = sys.getsizeof(value)
        self.__producer.send(topic=self.__topic, value=value)
        print(f"OK {self.counter} size (octets) {size}")
        self.counter += 1
        return message

    def on_completed(self):
        self.__producer.close()
        return super().on_completed()
