# deepgram_service.py
# Dedicated FastAPI service for Deepgram transcription.
# ====================================================

import logging
import sys
import os
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# CORRECTED IMPORTS for deepgram-sdk==5.0.0
from deepgram import DeepgramClient
from deepgram.options import LiveOptions, PrerecordedOptions

from pydub import AudioSegment
from typing import Optional

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING DEEPGRAM TRANSCRIPTION SERVICE (ON RENDER) ===")

# --- Initialization ---
# Load environment variables
load_dotenv()

# Deepgram API Key
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
logger.info(f"DEBUG: Environment variable 'DEEPGRAM_API_KEY' found: {bool(DEEPGRAM_API_KEY)}")

if not DEEPGRAM_API_KEY:
    logger.error("DEEPGRAM_API_KEY not configured. Deepgram service will not function.")

# Initialize Deepgram Client
deepgram_client = None
if DEEPGRAM_API_KEY:
    try:
        # Use api_key as a keyword argument to avoid positional argument issues
        deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        logger.info("Deepgram client initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Deepgram client: {e}")
        deepgram_client = None
else:
    logger.warning("Deepgram API key is missing, client will not be initialized.")

app = FastAPI(title="Deepgram Transcription Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---
def compress_audio_for_transcription(input_path: str, output_path: str = None) -> str:
    """Compress audio file optimally for transcription."""
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"
    
    try:
        logger.info(f"Compressing {input_path} for transcription...")
        audio = AudioSegment.from_file(input_path)
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio")
        
        target_sample_rate = 16000
        audio = audio.set_frame_rate(target_sample_rate)
        logger.info(f"Reduced sample rate to {target_sample_rate} Hz")
        
        audio.export(
            output_path, 
            format="mp3",
            bitrate="64k",
            # Ensure correct parameters for high-quality single-channel, 16kHz MP3
            parameters=["-q:a", "9", "-ac", "1", "-ar", str(target_sample_rate)]
        )
        logger.info(f"Audio compression complete: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error compressing audio: {e}")
        logger.warning(f"Compression failed for {input_path}, returning original.")
        return input_path

# --- FastAPI Endpoint ---
@app.post("/transcribe")
async def transcribe_audio_deepgram(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("en"),
    speaker_labels_enabled: bool = Form(False)
):
    logger.info(f"Deepgram transcription endpoint called for {file.filename}, lang: {language_code}, speakers: {speaker_labels_enabled}")

    if not deepgram_client:
        raise HTTPException(status_code=503, detail="Deepgram service is not initialized (API key missing).")

    # Save uploaded file temporarily
    tmp_path = ""
    compressed_path = ""
    try:
        # 1. Save original file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # 2. Compress audio (optional but good practice)
        compressed_path = compress_audio_for_transcription(tmp_path)

        # 3. Prepare transcription options using the dedicated Pydantic class
        # This is the modern, recommended way for deepgram-sdk==5.0.0
        options = PrerecordedOptions(
            model="nova-3",
            language=language_code,
            smart_format=True,
            punctuate=True,
            diarize=speaker_labels_enabled,
            utterances=speaker_labels_enabled
        )

        # 4. Transcribe using the FILE PATH method (takes 2 positional args: path, options)
        # The first argument is the file path, the second is the options object/dict
        # Using await asyncio.to_thread is correct for non-async SDK calls
        response = await asyncio.to_thread(
            deepgram_client.listen.v1.media.transcribe_file,
            compressed_path, # Positional Argument 1 (File Path)
            options          # Positional Argument 2 (Options Object/Dict)
        )

        # --- Process response ---
        response_dict = response.to_dict()
        transcript_text = ""
        has_speaker_labels = False
        
        if "results" in response_dict and "channels" in response_dict["results"] and response_dict["results"]["channels"]:
            # Check for paragraphs/utterances first if speaker labels are enabled
            if speaker_labels_enabled and "utterances" in response_dict["results"]:
                utterances = response_dict["results"]["utterances"]
                if utterances:
                    formatted_text = []
                    for utterance in utterances:
                        if isinstance(utterance, dict) and 'speaker' in utterance and 'transcript' in utterance:
                            # Speaker indices are 0-based, display as 1-based
                            speaker_num = utterance['speaker'] + 1
                            formatted_text.append(f"<strong>Speaker {speaker_num}:</strong> {utterance['transcript']}")
                    if formatted_text:
                        transcript_text = "\n".join(formatted_text)
                        has_speaker_labels = True
                else: 
                    # Fallback to full transcript if utterances are empty despite being requested
                    transcript_text = response_dict["results"]["channels"][0]["alternatives"][0].get("transcript", "")
            else: # If speaker labels not enabled, get full transcript
                transcript_text = response_dict["results"]["channels"][0]["alternatives"][0].get("transcript", "")

        logger.info(f"Deepgram transcription completed for {file.filename}")
        return {
            "status": "completed",
            "transcription": transcript_text,
            "language": language_code,
            "has_speaker_labels": has_speaker_labels,
            "service_used": "deepgram"
        }

    except Exception as e:
        logger.error(f"Deepgram transcription failed: {e}")
        # Re-raise as HTTPException for the client
        raise HTTPException(status_code=500, detail=f"Deepgram transcription failed: {str(e)}")
    finally:
        # Cleanup
        for path in [tmp_path, compressed_path]:
            if os.path.exists(path):
                os.unlink(path)
                logger.info(f"Cleaned up temp file: {path}")

@app.get("/")
async def root():
    return {"message": "Deepgram Transcription Service is running!"}