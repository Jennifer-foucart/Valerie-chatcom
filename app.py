import json
import base64
import io
import wave
import re
import os
import subprocess
import tempfile

import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
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
vosk_model = Model(os.path.join(_BASE_DIR, VOSK_MODEL_PATH))

conversation_history = []

inworld_session = requests.Session()
inworld_session.headers.update({
    "Authorization": f"Basic {INWORLD_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})

# =========================
# HELPERS
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


def synthesize_to_wav(text: str) -> bytes:
    """Convert a single sentence to WAV bytes via Inworld TTS."""
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
    with inworld_session.post(INWORLD_TTS_URL, json=payload, stream=True, timeout=15) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk_j   = json.loads(line)
            result    = chunk_j.get("result", {})
            audio_b64 = result.get("audioContent")
            if not audio_b64:
                continue
            pcm = extract_pcm_from_wav_chunk(base64.b64decode(audio_b64))
            if pcm:
                all_pcm += pcm
    return pcm_to_wav_bytes(all_pcm)


def split_sentences(text: str):
    """Split text into sentences on . ! ? … — keeping punctuation attached."""
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.webm")
        output_path = os.path.join(tmpdir, "output.wav")

        with open(input_path, "wb") as f:
            f.write(audio_bytes)

        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
            capture_output=True, check=True
        )

        rec = KaldiRecognizer(vosk_model, 16000)
        transcribed = []

        with open(output_path, "rb") as wf:
            wf.read(44)
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
    audio_bytes = request.data
    if not audio_bytes:
        return jsonify({"error": "No audio received"}), 400
    try:
        text = transcribe_audio(audio_bytes)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    1. Gets full reply from Mistral
    2. Splits into sentences
    3. Streams each sentence as a JSON line:
       {"sentence": "...", "audio": "<base64 WAV>"}
    Browser plays audio chunks in order as they arrive —
    speech starts after the first sentence, not the full reply.
    """
    data      = request.get_json()
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    conversation_history.append({"role": "user", "content": user_text})

    try:
        response        = client.beta.conversations.start(agent_id=AGENT_ID, inputs=conversation_history)
        assistant_reply = response.outputs[0].content
        conversation_history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sentences = split_sentences(assistant_reply)

    @stream_with_context
    def generate():
        for sentence in sentences:
            try:
                wav = synthesize_to_wav(sentence)
                audio_b64 = base64.b64encode(wav).decode("utf-8")
            except Exception:
                audio_b64 = ""

            line = json.dumps({"sentence": sentence, "audio": audio_b64})
            yield line + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


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