FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system deps (ffmpeg for pydub)
# Only the packages we actually need, and none of the "recommended" extras.
# Without --no-install-recommends this pulls in ~253MB including mesa and
# vulkan graphics drivers, which a transcription server never uses, and the
# extra download is what kept failing the build on a Debian mirror hiccup.
# Acquire::Retries rides out those hiccups instead of failing the whole build.
RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
    ffmpeg \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "deepgram_service:app", "--host", "0.0.0.0", "--port", "8000"]