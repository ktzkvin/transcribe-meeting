from pydub import AudioSegment
import pyaudio

from src.consumer.base import BaseConsumer


class FileAudioConsumer(BaseConsumer):
    def __init__(self, filename: str, chunk_duration: int):
        super().__init__()
        self.audio = AudioSegment.from_file(filename)
        self.chunk_duration = chunk_duration
        self.chunk_size = int(
            self.chunk_duration * 1000
        )  # Convert duration to milliseconds

    def consume(self) -> AudioSegment:

        for start in range(0, len(self.audio), self.chunk_size):
            yield self.audio[start: start + self.chunk_size]


class MicrophoneConsumer(BaseConsumer):
    def __init__(self, chunk_duration: int, rate: int = 44100):
        super().__init__()
        self.chunk_duration = chunk_duration
        self.rate = rate
        self.channels = 1
        self.format = pyaudio.paInt16  # 16 bits per sample
        self.chunk_size = int(
            self.rate * self.chunk_duration
        )  # Calculate chunk size based on duration
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    def consume(self) -> AudioSegment:
        try:
            while True:
                data = self.stream.read(self.chunk_size)
                audio_segment = AudioSegment(
                    data=data,
                    sample_width=self.p.get_sample_size(self.format),
                    frame_rate=self.rate,
                    channels=self.channels,
                )
                yield audio_segment
        except KeyboardInterrupt:
            print("Recording stopped.")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    # File consumer example
    file_consumer = FileAudioConsumer("data/sample-3s.wav", chunk_duration=1)
    for audio_chunk in file_consumer.consume():
        print(audio_chunk)
    print("File audio chunk loaded.")

    # Microphone consumer example
    mic_consumer = MicrophoneConsumer(chunk_duration=3)
    print("Microphone audio chunks:")
    for mic_chunk in mic_consumer.consume():
        print(f"Received chunk of size {len(mic_chunk)} bytes.")
        exit()
