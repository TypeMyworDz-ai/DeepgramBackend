# DeepgramBackend

Standalone Deepgram transcription service for TypeMyworDz AI.

It is called by the main backend as the last fallback in the transcription
chain, and is the only engine used by the dedicated Deepgram test account.

## Endpoints

- `GET /` - liveness check.
- `POST /transcribe` - multipart form. Fields: `file`, `language_code`,
  `speaker_labels_enabled`. Returns `status`, `transcription`, `language`,
  `has_speaker_labels`, `service_used`.

## Configuration

- `DEEPGRAM_API_KEY` - required.
- Listens on port 8000 (`EXPOSE 8000`, entry point `deepgram_service:app`).

## Notes on audio handling

Audio is sent to Deepgram as-is when the file is under 20 MB. Larger files are
downmixed to mono and re-encoded at 64 kbit/s. Two rules matter here:

1. Never pass a quality flag alongside an explicit bitrate. The encoder honours
   the quality flag and silently ignores the bitrate, which previously crushed
   speech to about 10 kbit/s and produced near-empty transcripts.
2. Never raise the sample rate above the source. Upsampling adds no detail and
   only makes the upload larger.
