# FROM python:3.9-slim
FROM python:3.9-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y ffmpeg portaudio19-dev libsndfile1 
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app


RUN uv sync

CMD ["bash"]
