# deepgram_service.py
# Dedicated FastAPI service for Deepgram transcription.
# ====================================================

import logging
import sys
import os
import functools
import tempfile
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# CORRECTED IMPORTS for deepgram-sdk==3.11.0
from deepgram import DeepgramClient, PrerecordedOptions

from pydub import AudioSegment
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING DEEPGRAM TRANSCRIPTION SERVICE ===")

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

# Audio at or below this size is sent to Deepgram exactly as the client uploaded
# it. Deepgram accepts files far larger than this, so shrinking a file this small
# saves no meaningful upload time, and because MP3 is a lossy format every
# re-encode permanently throws away a little more of the speech. Compression is
# therefore only worth doing to keep genuinely large uploads moving.
COMPRESS_ABOVE_BYTES = 20 * 1024 * 1024  # 20 MB

# Speech carries almost all of its intelligibility below 8 kHz, so 16 kHz is the
# standard sample rate for transcription. We never raise a file above this, and
# just as importantly we never raise a file that arrived lower: upsampling an
# 8 kHz phone recording to 16 kHz invents no detail, it only makes the file
# bigger. So this is a ceiling, not a target.
MAX_SAMPLE_RATE = 16000

# The container type we tell Deepgram to expect. Deepgram inspects the audio
# itself, but sending the right type avoids it having to guess.
MIME_BY_EXTENSION = {
    ".mp3": "audio/mpeg", ".mpga": "audio/mpeg", ".mp2": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp4": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".oga": "audio/ogg",
    ".opus": "audio/ogg", ".flac": "audio/flac", ".webm": "audio/webm",
    ".amr": "audio/amr", ".3gp": "audio/3gpp", ".wma": "audio/x-ms-wma",
    ".aiff": "audio/aiff", ".aif": "audio/aiff", ".caf": "audio/x-caf",
}


def mimetype_for(path: str) -> str:
    """Best guess at the audio type from the file name, defaulting to MP3."""
    return MIME_BY_EXTENSION.get(os.path.splitext(path)[1].lower(), "audio/mpeg")


def compress_audio_for_transcription(input_path: str, output_path: str = None) -> str:
    """Shrink a large upload before sending it to Deepgram.

    Returns the path of the file that should actually be sent, which is the
    original file whenever compressing it would not help. Never raises: if
    anything goes wrong we send the original audio rather than fail the job.
    """
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_compressed.mp3"

    try:
        size = os.path.getsize(input_path)
        if size <= COMPRESS_ABOVE_BYTES:
            logger.info(
                f"Sending audio as uploaded ({size} bytes); re-encoding a file "
                f"this small would lose quality for no useful saving."
            )
            return input_path

        logger.info(f"Compressing {input_path} for transcription ({size} bytes)...")
        audio = AudioSegment.from_file(input_path)

        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted to mono audio")

        target_sample_rate = min(MAX_SAMPLE_RATE, audio.frame_rate)
        if target_sample_rate != audio.frame_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            logger.info(f"Reduced sample rate to {target_sample_rate} Hz")
        else:
            logger.info(f"Keeping sample rate at {target_sample_rate} Hz")

        # Note on the bitrate: do NOT add "-q:a" here. Passing a quality setting
        # alongside a bitrate makes the encoder ignore the bitrate entirely and
        # use the quality scale instead. This code used to pass "-q:a", "9",
        # which is the very lowest quality the encoder offers, and the result was
        # that audio meant to be encoded at 64 kbit/s came out at roughly
        # 10 kbit/s. Speech that badly degraded is close to unintelligible, and
        # Deepgram was returning almost empty transcripts because of it.
        audio.export(
            output_path,
            format="mp3",
            bitrate="64k",
            parameters=["-ac", "1", "-ar", str(target_sample_rate)]
        )
        logger.info(
            f"Audio compression complete: {output_path} "
            f"({os.path.getsize(output_path)} bytes)"
        )
        return output_path

    except Exception as e:
        logger.error(f"Error compressing audio: {e}")
        logger.warning(f"Compression failed for {input_path}, returning original.")
        return input_path

# Deepgram's current best general-purpose model. It is a straight upgrade on
# nova-2 for accuracy on noisy, multi-speaker and far-field audio, which is
# exactly the sort of recording our clients send. nova-2 stays as the fallback
# for the handful of languages nova-3 does not cover yet.
BEST_MODEL = "nova-3"
FALLBACK_MODEL = "nova-2"

# Languages nova-3 handles. Anything outside this list goes to nova-2.
NOVA_3_LANGUAGES = {
    "multi", "af", "af-ZA", "ar", "ar-AE", "ar-SA", "ar-QA", "ar-KW", "ar-SY",
    "ar-LB", "ar-PS", "ar-JO", "ar-EG", "ar-SD", "ar-TD", "ar-MA", "ar-DZ",
    "ar-TN", "ar-IQ", "ar-IR", "hy", "as", "as-IN", "be", "bn", "bs", "bg",
    "ca", "zh-HK", "zh", "zh-CN", "zh-Hans", "zh-TW", "zh-Hant", "hr", "cs",
    "cs-CZ", "da", "da-DK", "nl", "en", "en-US", "en-AU", "en-GB", "en-IN",
    "en-NZ", "et", "fi", "nl-BE", "fr", "fr-CA", "ka", "ka-GE", "de", "de-CH",
    "el", "gu", "gu-IN", "he", "hi", "hu", "id", "it", "ja", "kn", "kk",
    "kk-KZ", "ko", "ko-KR", "lv", "lt", "mk", "ms", "mr", "mn", "ne", "no",
    "ps", "ps-AF", "fa", "pl", "pt", "pt-BR", "pt-PT", "pa", "pa-IN", "ro",
    "ru", "sr", "sk", "sl", "es", "es-419", "sv", "sv-SE", "tl", "ta", "te",
    "th", "th-TH", "tr", "tr-TR", "uk", "ur", "vi",
}


def model_for_language(language_code: Optional[str]) -> str:
    """Pick the best Deepgram model that supports the requested language."""
    code = (language_code or "en").strip()
    if code in NOVA_3_LANGUAGES or code.split("-")[0] in NOVA_3_LANGUAGES:
        return BEST_MODEL
    logger.info(f"Language {code} is not supported by {BEST_MODEL}; using {FALLBACK_MODEL}.")
    return FALLBACK_MODEL


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
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    compressed_path = tmp_path

    try:
        # Compress audio
        compressed_path = compress_audio_for_transcription(tmp_path)

        # Prepare audio buffer
        with open(compressed_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        # Payload (audio data)
        payload = {
            "buffer": buffer_data,
            "mimetype": mimetype_for(compressed_path)
        }

        # Transcription options (use PrerecordedOptions for type safety, or dict)
        options = PrerecordedOptions(
            model=model_for_language(language_code),
            language=language_code,
            smart_format=True,
            punctuate=True,
            diarize=speaker_labels_enabled,
            utterances=speaker_labels_enabled
        )

        # Opt every request out of Deepgram's Model Improvement Program so that
        # client audio and transcripts are never used to train Deepgram's models.
        # Deepgram treats this as a query parameter; the SDK exposes unsupported
        # query parameters through "addons".
        # https://developers.deepgram.com/docs/the-deepgram-model-improvement-partnership-program
        privacy_addons = {"mip_opt_out": "true"}

        # Transcribe (correct namespace and two args: payload, options)
        response = await asyncio.to_thread(
            functools.partial(
                deepgram_client.listen.prerecorded.v("1").transcribe_file,
                payload,
                options,
                addons=privacy_addons,
            )
        )

        # Process response with enhanced speaker handling
        response_dict = response.to_dict()
        transcript_text = ""
        has_speaker_labels = False
        
        if "results" in response_dict and "channels" in response_dict["results"] and response_dict["results"]["channels"]:
            alternative = response_dict["results"]["channels"][0]["alternatives"][0]
            
            if speaker_labels_enabled:
                # Deepgram returns utterances alongside the channels rather than
                # inside an alternative, so look in both places. Reading only the
                # alternative meant this branch never ran and every diarized job
                # quietly fell through to the rougher word-by-word path below.
                utterances = (
                    response_dict.get("results", {}).get("utterances")
                    or alternative.get("utterances")
                )
                if utterances:
                    formatted_text = []
                    for utterance in utterances:
                        if 'speaker' in utterance and 'transcript' in utterance:
                            speaker_num = utterance['speaker'] + 1
                            formatted_text.append(f"<strong>Speaker {speaker_num}:</strong> {utterance['transcript']}")
                    if formatted_text:
                        transcript_text = "\n".join(formatted_text)
                        has_speaker_labels = True
                elif "words" in alternative and alternative["words"]:  # Fallback to word-level diarization
                    words = alternative["words"]
                    formatted_text = []
                    current_speaker = None
                    current_line = []
                    for word in words:
                        if 'speaker' in word:
                            speaker = word['speaker'] + 1
                            if speaker != current_speaker:
                                if current_line:
                                    formatted_text.append(f"<strong>Speaker {current_speaker}:</strong> {' '.join(current_line)}")
                                    current_line = []
                                current_speaker = speaker
                            current_line.append(word.get('punctuated_word', word.get('word', '')))
                    if current_line:  # Add the last line
                        formatted_text.append(f"<strong>Speaker {current_speaker}:</strong> {' '.join(current_line)}")
                    if formatted_text:
                        transcript_text = "\n".join(formatted_text)
                        has_speaker_labels = True
            
            # Fallback to full transcript if no speaker labels processed
            if not transcript_text:
                transcript_text = alternative.get("transcript", "")

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
