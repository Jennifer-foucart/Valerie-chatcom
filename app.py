import json
import base64
import io
import wave
import threading
import os
import subprocess
import tempfile

import requests
from flask import Flask, render_template, request, jsonify, Response
from mistralai import Mistral
from vosk import Model, KaldiRecognizer

# =========================
# CONFIG
# =========================
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "92sp6C59uxNWYZFCpMKdisQYpdKhTD7i")
AGENT_ID        = os.environ.get("AGENT_ID", "ag_019c27dccf687399bf821ea5757ef36a")

INWORLD_API_KEY = os.environ.get("INWORLD_API_KEY", "OTdHdE1Hb0VseVM3RXhMVlNLYVFDMGcwOEZJbVF0eUY6OGhndjNhR3JhT0JyUXJqUWZWVXZqeWlTSFJRMDZSR3RTcllVRm9BS2VYUGFrTE9RTnpOQ0xteGlicTBzZGV3MQ==")
INWORLD_TTS_URL = "https://api.inworld.ai/tts/v1/voice:stream"

VOICE_ID = "Hélène"
MODEL_ID  = "inworld-tts-1.5-max"

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-fr-0.22")

# =========================
# INIT
# =========================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static")
)
client = Mistral(api_key=MISTRAL_API_KEY)

# Load Vosk model once at startup — same as original
vosk_model = Model(os.path.join(_BASE_DIR, VOSK_MODEL_PATH))

conversation_history = []

inworld_session = requests.Session()
inworld_session.headers.update({
    "Authorization": f"Basic {INWORLD_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})

# =========================
# TTS HELPERS  (unchanged from original)
# =========================

def extract_pcm_from_wav_chunk(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes())


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate=48000, channels=1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def synthesize_speech(text: str) -> bytes:
    payload = {
        "text": text,
        "voiceId": VOICE_ID,
        "modelId": MODEL_ID,
        "temperature": 1.48,
        "audio_config": {
            "audio_encoding": "LINEAR16",
            "sample_rate_hz": 48000,
            "speaking_rate": 1.1
        }
    }
    all_pcm = b""
    with inworld_session.post(INWORLD_TTS_URL, json=payload, stream=True, timeout=15) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk_j   = json.loads(line)
            result    = chunk_j.get("result", {})
            audio_b64 = result.get("audioContent")
            if not audio_b64:
                continue
            wav_bytes = base64.b64decode(audio_b64)
            pcm_bytes = extract_pcm_from_wav_chunk(wav_bytes)
            if pcm_bytes:
                all_pcm += pcm_bytes
    return pcm_to_wav_bytes(all_pcm)


# =========================
# VOSK TRANSCRIPTION  (same logic as original process_audio)
# =========================

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Receives a complete audio recording from the browser (webm/wav),
    converts to 16kHz mono WAV with ffmpeg — exactly like the original
    resampling step — then runs through Vosk KaldiRecognizer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.webm")
        output_path = os.path.join(tmpdir, "output.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        # Convert to 16kHz mono WAV — same target as original resample to 16000
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
            capture_output=True, check=True
        )

        # Feed through Vosk — identical to original recognizer.AcceptWaveform loop
        rec = KaldiRecognizer(vosk_model, 16000)
        transcribed = []

        with open(output_path, "rb") as wf:
            wf.read(44)  # skip WAV header
            while True:
                block = wf.read(4000)
                if not block:
                    break
                if rec.AcceptWaveform(block):
                    result = json.loads(rec.Result())
                    if result.get("text"):
                        transcribed.append(result["text"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            transcribed.append(final["text"])

        return " ".join(transcribed).strip()


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Browser sends a complete audio recording.
    Server transcribes with Vosk and returns the text.
    """
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"error": "No audio received"}), 400
    try:
        text = transcribe_audio(audio_bytes)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data      = request.get_json()
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_text})

    try:
        response        = client.beta.conversations.start(agent_id=AGENT_ID, inputs=conversation_history)
        assistant_reply = response.outputs[0].content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return jsonify({"reply": assistant_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        wav_bytes = synthesize_speech(text)
        return Response(
            wav_bytes,
            mimetype="audio/wav",
            headers={"Content-Disposition": "inline; filename=speech.wav"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(conversation_history)


@app.route("/reset", methods=["POST"])
def reset_history():
    conversation_history.clear()
    return jsonify({"status": "reset"})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)