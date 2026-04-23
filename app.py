import json
import base64
import io
import wave
import re
import os
import subprocess
import tempfile
import time

import numpy as np
import faiss
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



# Map of interview choice keys → RAG query
INTERVIEW_TYPES = {
    "motivational": "How the model should behave in case of a motivational interview?",
    "behavioral":   "How the model should behave in case of a behavioral interview?",
    "technical":    "How the model should behave in case of a technical interview?",
    # Add more as needed
}

# =========================
# INIT
# =========================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static")
)

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
vosk_model     = Model(os.path.join(_BASE_DIR, VOSK_MODEL_PATH))

# Per-session state: keyed by session_id sent from the browser
# Each entry: { "history": [...], "ready": bool }
sessions = {}

inworld_session = requests.Session()
inworld_session.headers.update({
    "Authorization": f"Basic {INWORLD_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})


# =========================
# RAG
# =========================

def get_text_embedding(text: str) -> list:
    resp = mistral_client.embeddings.create(model="mistral-embed", inputs=text)
    return resp.data[0].embedding


def build_rag_index(filepath="Modules.txt"):
    chunks = []
    current_chunk = []
    with open(os.path.join(_BASE_DIR, filepath), "r", encoding="utf-8") as f:
        for line in f:
            if re.match(r'^\d+\.\s+', line):
                if current_chunk:
                    chunks.append("".join(current_chunk).strip())
                    current_chunk = []
            current_chunk.append(line)
    if current_chunk:
        chunks.append("".join(current_chunk).strip())

    embeddings = []
    for chunk in chunks:
        embeddings.append(get_text_embedding(chunk))
        time.sleep(1)
    embeddings = np.array(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index


def retrieve_system_prompt(question: str, k: int = 1) -> str:
    q_emb = np.array([get_text_embedding(question)])
    _, I  = _rag_index.search(q_emb, k=k)
    retrieved = [_rag_chunks[i] for i in I.tolist()[0]]
    return "\n\n".join(retrieved)


# Build index once at startup
print("Building RAG index…")
_rag_chunks, _rag_index = build_rag_index("Modules.txt")
print(f"RAG index ready — {len(_rag_chunks)} chunks.")


# =========================
# HELPERS
# =========================

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def stream_opus_chunks(text: str):
    payload = {
        "text": text,
        "voiceId": VOICE_ID,
        "modelId": MODEL_ID,
        "temperature": 1.48,
        "audio_config": {
            "audio_encoding": "OGG_OPUS",
            "sample_rate_hz": 48000,
            "speaking_rate": 1.1
        }
    }
    with inworld_session.post(INWORLD_TTS_URL, json=payload, stream=True, timeout=15) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk_j   = json.loads(line)
            result    = chunk_j.get("result", {})
            audio_b64 = result.get("audioContent")
            if audio_b64:
                yield base64.b64decode(audio_b64)


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
                    r = json.loads(rec.Result())
                    if r.get("text"):
                        transcribed.append(r["text"])

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


@app.route("/start_session", methods=["POST"])
def start_session():
    """
    Called when the user clicks an interview type button.
    Body: { "session_id": "abc123", "interview_type": "motivational" }
    Runs RAG for the chosen type, initialises conversation history with
    the retrieved system prompt, and returns 200 when ready.
    """
    data           = request.get_json()
    session_id     = data.get("session_id", "").strip()
    interview_type = data.get("interview_type", "").strip()

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if interview_type not in INTERVIEW_TYPES:
        return jsonify({"error": f"Unknown interview_type '{interview_type}'",
                        "valid": list(INTERVIEW_TYPES.keys())}), 400

    question      = INTERVIEW_TYPES[interview_type]
    system_prompt = retrieve_system_prompt(question)

    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "interview_type": interview_type,
        "ready": True,
    }

    return jsonify({"status": "ready", "interview_type": interview_type})


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
    import traceback
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()
    user_text  = data.get("message", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found — call /start_session first"}), 400
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    history = sessions[session_id]["history"]
    history.append({"role": "user", "content": user_text})

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=history,
        )
        assistant_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500
    """
    Body: { "session_id": "abc123", "message": "user text" }
    Requires /start_session to have been called first.
    """
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()
    user_text  = data.get("message", "").strip()

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found — call /start_session first"}), 400
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    history = sessions[session_id]["history"]
    history.append({"role": "user", "content": user_text})

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=history,
        )
        assistant_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    sentences = split_sentences(assistant_reply)

    @stream_with_context
    def generate():
        for sentence in sentences:
            yield json.dumps({"type": "sentence_start", "text": sentence}) + "\n"
            try:
                for opus_chunk in stream_opus_chunks(sentence):
                    yield json.dumps({
                        "type": "audio",
                        "data": base64.b64encode(opus_chunk).decode()
                    }) + "\n"
            except Exception:
                pass
            yield json.dumps({"type": "sentence_end"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/history", methods=["GET"])
def get_history():
    session_id = request.args.get("session_id", "")
    if session_id not in sessions:
        return jsonify([])
    # Strip the system message before returning to the browser
    return jsonify([m for m in sessions[session_id]["history"] if m["role"] != "system"])


@app.route("/reset", methods=["POST"])
def reset_history():
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "reset"})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
