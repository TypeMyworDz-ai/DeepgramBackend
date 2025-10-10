FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system deps (ffmpeg for pydub)
RUN apt-get update && apt-get install -y \
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