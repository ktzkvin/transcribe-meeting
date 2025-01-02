### Kafka Audio Producer and Consumer Setup

To set up and run the Kafka Audio Producer and Consumer for audio processing, follow the steps below. This will allow you to produce and consume audio data to/from Kafka, as well as transcribe the audio using the Whisper model.

#### 1. **Start Kafka with Docker Compose**

To start Kafka using Docker Compose, run the following command:

```bash
docker-compose up
```

This will start the Kafka service along with its dependencies (like Zookeeper). Ensure that Docker and Docker Compose are installed and configured on your system.

#### 2. **Install Required Packages**

To install all the required packages for running the scripts, run the following command:

```bash
uv sync
```

This will install all the dependencies specified in your project, ensuring that everything is ready for the producer and consumer to work.

#### 3. **Running the Audio Kafka Producer**

To produce audio into Kafka from a microphone or a file, you can use the `audio_kafka_producer.py` script. The usage for this script is as follows:

```bash
usage: audio_kafka_producer.py [-h] [--topic TOPIC] [--bootstrap-servers BOOTSTRAP_SERVERS] [--source {microphone,file}] [--block-duration BLOCK_DURATION] [--device DEVICE] [--file-path FILE_PATH]
```

##### **Arguments**:

- `-h, --help`: Show the help message and exit.
- `--topic TOPIC`: Name of the Kafka topic to send audio data to (default: `audio`).
- `--bootstrap-servers BOOTSTRAP_SERVERS`: Kafka bootstrap servers to connect to (default: `localhost:29092`).
- `--source {microphone,file}`: Choose the audio source (`microphone` or `file`). Default is `microphone`.
- `--block-duration BLOCK_DURATION`: Duration in seconds for reading audio blocks (default: `2`).
- `--device DEVICE`: Microphone device ID to use (default: first available).
- `--file-path FILE_PATH`: Path to the audio file if `source` is `file`.

##### **Example Command**:

To produce audio from a microphone and send it to Kafka:

```bash
uv run audio_kafka_producer.py --source microphone --block-duration 2 --topic audio
```

To produce audio from a file:

```bash
uv run audio_kafka_producer.py --source file --file-path "path/to/audio/file.wav" --topic audio
```

---

#### 4. **Consuming Audio Messages from Kafka**

To consume messages from the Kafka `audio` topic, you can use the `audio_kafka_consumer_debug.py` script. The usage for this script is:

```bash
usage: audio_kafka_consumer_debug.py [-h] [--topic TOPIC] [--bootstrap-servers BOOTSTRAP_SERVERS] [--sample-rate SAMPLE_RATE] [--chunk-size CHUNK_SIZE] [--log-file LOG_FILE] [--write-data]
```

##### **Arguments**:
- `-h, --help`: Show the help message and exit.
- `--topic TOPIC`: Kafka topic to consume from (default: `audio`).
- `--bootstrap-servers BOOTSTRAP_SERVERS`: Kafka bootstrap servers to connect to (default: `localhost:29092`).
- `--sample-rate SAMPLE_RATE`: Sample rate for the audio data (default: `16000`).
- `--chunk-size CHUNK_SIZE`: Size of each audio chunk (default: `5`).
- `--log-file LOG_FILE`: Path to save the debug log output (default: `audio-debug.log`).
- `--write-data`: Whether to log audio data (default: `False`).

##### **Example Command**:
To consume messages from the `audio` topic and log debug output:
```bash
uv run audio_kafka_consumer_debug.py --topic audio --bootstrap-servers localhost:29092 --log-file audio-debug.log --write-data
```

---

#### 5. **Transcribing Audio from Kafka with Whisper**
To consume and transcribe the audio from Kafka using the Whisper model, you can use the `audio_kafka_consumer_whisper.py` script. The usage for this script is:

```bash
usage: audio_kafka_consumer_whisper.py [-h] [--topic TOPIC] [--bootstrap-servers BOOTSTRAP_SERVERS] [--sample-rate SAMPLE_RATE] [--chunk-size CHUNK_SIZE] [--log-file LOG_FILE] [--write-data] [--model {tiny,base,small,medium,large}] [--device {cuda,cpu}]
```

##### **Arguments**:
- `-h, --help`: Show the help message and exit.
- `--topic TOPIC`: Kafka topic to consume from (default: `audio`).
- `--bootstrap-servers BOOTSTRAP_SERVERS`: Kafka bootstrap servers to connect to (default: `localhost:29092`).
- `--sample-rate SAMPLE_RATE`: Sample rate for the audio data (default: `16000`).
- `--chunk-size CHUNK_SIZE`: Size of each audio chunk (default: `5`).
- `--log-file LOG_FILE`: Path to save the debug log output (default: `audio-debug.log`).
- `--write-data`: Whether to log audio data (default: `False`).
- `--model {tiny,base,small,medium,large}`: Whisper model size to use for transcription (default: `small`).
- `--device {cuda,cpu}`: Device to use for Whisper transcription (`cuda` for GPU, `cpu` for CPU, default: `cuda`).

##### **Example Command**:
To consume and transcribe audio messages from Kafka:
```bash
uv run audio_kafka_consumer_whisper.py --topic audio --bootstrap-servers localhost:29092 --log-file transcription-log.log --model large --device cuda
```

This command will consume audio from the `audio` Kafka topic, transcribe the audio using the `large` Whisper model on a GPU (`cuda`), and log the transcription results to `transcription-log.log`.

