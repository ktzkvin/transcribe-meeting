
# Installation

Requirement :
- `uv` ([installation](https://docs.astral.sh/uv/getting-started/installation/)) 
- ```bash
    sudo apt-get install ffmpeg portaudio19-dev libsndfile1
    ```
- ```bash
    uv sync
    ``` 

## Docker 
Build 
```bash
docker build -t diart-uv-app .
```
Run  (with audio file define)
```bash
docker run -it -v /path/to/local/config.json:/app/config/config.json diart-uv-app uv run /app/src/cli/diarization.py
```
Run (With mic )
```bash
docker run -it --device /dev/snd:/dev/snd -v /path/to/local/config.json:/app/config/config.json diart-uv-app uv run /app/src/cli/diarization.py
```

example of config:
```json
{
    "segmentation_model": "pyannote/segmentation-3.0",
    "embedding_model": "pyannote/wespeaker-voxceleb-resnet34-LM",
    "mic": null,
    "audio_path": null,  
    "device": "cuda",
    "sample_rate" : 16000
  }
  
```