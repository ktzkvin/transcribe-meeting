from fastapi import FastAPI, File, UploadFile, Header, Form
from fastapi.responses import JSONResponse
import torch
import io
import soundfile as sf
import numpy as np
from dataclasses import asdict

from time import time
from src.transcriber.whisper import WhisperTranscriber
from src.schema.audio import AudioFileMetadata, AudioMetadataInput
from src.schema.transcription import TranscriptionOutput

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_version = "tiny"
api_version = "0.0.1"
transcription_model = WhisperTranscriber(model=model_version, device=device)


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    model: str = Form(...),
    language: str = Form(None),
    prompt: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0),
    timestamp_granularities: list[str] = Form(None),
) -> TranscriptionOutput:

    # Read the audio file
    t_read = time()
    audio_bytes = await file.read()
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
    t_read = time() - t_read

    t_infer = time()
    # Ensure the audio is mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    aduio_metadata = AudioFileMetadata(
        audio_data=audio,
        start_time=0,
        end_time=30,
        audio_segment_index=0,
        sample_rate=sample_rate,
        source=file.filename,
    )
    response_data = transcription_model(aduio_metadata)
    response_data = asdict(response_data)
    t_infer = time() - t_infer
    response_data["metadata"] = {
        "inference_rime": t_infer,
        "read_time": t_read,
        "model_version": model_version,
    }

    return JSONResponse(response_data)


@app.post("/v1/audio/transcriptions/metadata")
async def transcribe_metadata(
    audio: AudioMetadataInput,
) -> TranscriptionOutput:
    t_read = time()
    audio_metadata = AudioFileMetadata.from_dict(audio.model_dump())
    t_read = time() - t_read

    t_infer = time()
    response_data = transcription_model(audio_metadata)
    response_data = asdict(response_data)
    t_infer = time() - t_infer
    response_data["metadata"] = {
        "inference_rime": t_infer,
        "read_time": t_read,
        "model_version": model_version,
    }

    return JSONResponse(response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
