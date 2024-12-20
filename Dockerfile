FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.astral.sh/uv | bash

ENV PATH="$PATH:/root/.local/bin"

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install diart huggingface_hub pyannote.audio

COPY . /app


RUN uv sync

CMD ["bash"]
