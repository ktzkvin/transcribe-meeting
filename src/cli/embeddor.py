from src.consumer.kafka import KafkaCastConsumer
from src.schema.config.consumer import KafkaConsumerConfig
from src.schema.segmentation import SpeakerSegment
from src.diarization.embedding import AudioEmbedding
from src.schema.segmentation import SpeakerSegment

if __name__ == "__main__":
    consumer = KafkaCastConsumer(
        kafka_config=KafkaConsumerConfig(
            topic="segmentation-topic",
            bootstrap_servers="localhost:29092",
            fetch_max_bytes=16_000_000,
            enable_auto_commit=True,
            groud_id="test",
            auto_offset_reset="latest",
        ),
        dataclass=SpeakerSegment,
    )
    embedding = AudioEmbedding()

    while True:
        for item in consumer.consume():
            if isinstance(item, list):
                emb = embedding(item)
            else:
                emb = embedding([item])
