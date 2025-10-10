# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for Deepgram and general Python compilation
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Upgrade pip, setuptools, and wheel first to ensure a robust installation environment
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install any needed Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Deepgram service code into the container
COPY deepgram_service.py .

# Command to run your Deepgram service using Gunicorn for the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "deepgram_service:app"]
