FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md ./
COPY upscaler ./upscaler

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

ENTRYPOINT ["upscaler"]
