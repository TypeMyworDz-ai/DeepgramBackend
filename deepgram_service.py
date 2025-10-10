import os
from flask import Flask, request, jsonify
from deepgram import DeepgramClient, LiveTranscriptionEvents, DeepgramClientOptions, Microphone

app = Flask(__name__)

# Initialize Deepgram Client
# Make sure to set your Deepgram API key as an environment variable in Render
# e.g., DEEPGRAM_API_KEY="YOUR_API_KEY"
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("Warning: DEEPGRAM_API_KEY environment variable not set. Deepgram functionality may not work.")
    # Optionally, raise an error or handle this more robustly in production
    # raise ValueError("DEEPGRAM_API_KEY environment variable not set.")

# Function to transcribe a pre-recorded audio file using Deepgram
def transcribe_audio_file_with_deepgram(audio_buffer):
    if not DEEPGRAM_API_KEY:
        return "Deepgram API key not configured.", None

    client = DeepgramClient(DEEPGRAM_API_KEY)
    payload = {"buffer": audio_buffer}

    try:
        response = client.listen.prerecorded.v("1").transcribe_file(payload, {"punctuate": True})
        transcript = response.results.channels[0].alternatives[0].transcript
        return None, transcript
    except Exception as e:
        print(f"Deepgram API Error: {e}")
        return f"Deepgram API Error: {e}", None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_buffer = audio_file.read()

    error_message, transcript = transcribe_audio_file_with_deepgram(audio_buffer)

    if error_message:
        return jsonify({"error": error_message}), 500
    if transcript:
        return jsonify({"transcript": transcript})
    return jsonify({"error": "Transcription failed for an unknown reason"}), 500

@app.route('/')
def health_check():
    return "Deepgram service is running."

if __name__ == '__main__':
    # For local development, you can run it directly:
    # app.run(host='0.0.0.0', port=8000, debug=True)
    # For production on Render, Gunicorn will handle running the app.
    print("Deepgram service is ready to be run by Gunicorn or locally.")
