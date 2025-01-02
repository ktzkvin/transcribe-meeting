import time
import traceback
import rx.operators as ops
from src.sources.audio import FileAudioSourceTimed

from src.transcriber.whisper import WhisperTranscriber


def main(
    filename: str = "data/audios/audio.wav",
    model: str = "small",
    block_duration: int = 30,
    sample_rate: int = 16_000,
    device: str = "cuda",
):
    source = FileAudioSourceTimed(
        file=filename,
        sample_rate=sample_rate,
        block_duration=block_duration,
    )

    asr = WhisperTranscriber(model=model, device=device)

    source.stream.pipe(ops.map(asr), ops.map(print)).subscribe(
        on_error=lambda _: traceback.print_exc()
    )
    source.read()


if __name__ == "__main__":
    from glob import glob
    import soundfile as sf
    import pandas as pd
    import GPUtil

    models = [
        "tiny",
        "small",
        "base",
        "medium",
        "turbo",
        "large",
    ]
    data = []

    for model in models:
        for device in ["cuda"]:
            for duration in [15]:
                for file in glob("data/**/*.wav"):
                    gpus = GPUtil.getGPUs()
                    f = sf.SoundFile(file)
                    seconds = f.frames / f.samplerate
                    f.close()

                    tmp = {
                        "filename": file,
                        "model_name": model,
                        "device": device,
                        "block_duration": duration,
                        "total_duration": seconds,
                        "gpu.name": [gpu.name for gpu in gpus][
                            0
                        ],  # TODO: Select the right gpu
                        "gpu.memoryUse": [gpu.memoryUsed for gpu in gpus][
                            0
                        ],  # TODO: Select the right gpu
                    }
                    t_start = time.time()
                    main(
                        filename=file,
                        model=model,
                        block_duration=duration,
                        device=device,
                    )
                    t_end = time.time()
                    tmp["process_time"] = t_end - t_start
                    data.append(tmp)

                df = pd.DataFrame(data)
                df["process_time_per_second"] = (
                    df["process_time"] / df["total_duration"]
                )
                df.to_csv("data/raw_stats.csv", index=False)
                avg_process_time_by_model = df.groupby(["model_name", "gpu.name"])[
                    "process_time_per_second"
                ].mean()
                avg_process_time_df = avg_process_time_by_model.reset_index()
                avg_process_time_df.columns = [
                    "model_name",
                    "gpu.name",
                    "avg_process_time_per_second",
                ]

                avg_process_time_df.to_csv(
                    "data/avg_process_time_per_duration_by_model.csv", index=False
                )

                print(avg_process_time_df)
