import time
import traceback
import rx.operators as ops
from src.sources.audio import FileAudioSourceTimed

from src.transcriber.whisper import WhisperApiTranscriber


def main(
    filename: str = "data/audios/audio.wav",
    model: str = "small",
    block_duration: int = 30,
    sample_rate: int = 16_000,
):
    source = FileAudioSourceTimed(
        file=filename,
        sample_rate=sample_rate,
        block_duration=block_duration,
    )

    asr = WhisperApiTranscriber(api_url=model)

    source.stream.pipe(ops.map(asr)).subscribe(on_error=lambda _: traceback.print_exc())
    source.read()


if __name__ == "__main__":
    from glob import glob
    import soundfile as sf
    import pandas as pd

    models = ["http://localhost:8000/v1/audio/transcriptions/metadata"]
    data = []

    for model in models:
        for duration in [15]:
            for file in glob("data/**/*.wav"):
                f = sf.SoundFile(file)
                seconds = f.frames / f.samplerate
                f.close()

                tmp = {
                    "filename": file,
                    "api_url": model,
                    "block_duration": duration,
                    "total_duration": seconds,
                }
                t_start = time.time()
                main(
                    filename=file,
                    model=model,
                    block_duration=duration,
                )
                t_end = time.time()
                tmp["process_time"] = t_end - t_start
                data.append(tmp)

            df = pd.DataFrame(data)
            df["process_time_per_second"] = df["process_time"] / df["total_duration"]
            df.to_csv("data/api_raw_stats.csv", index=False)
            avg_process_time_by_model = df.groupby(
                [
                    "api_url",
                ]
            )["process_time_per_second"].mean()
            avg_process_time_df = avg_process_time_by_model.reset_index()
            avg_process_time_df.columns = [
                "api_url",
                "avg_process_time_per_second",
            ]

            avg_process_time_df.to_csv(
                "data/api_avg_process_time_per_duration_by_model.csv", index=False
            )

            print(avg_process_time_df)
