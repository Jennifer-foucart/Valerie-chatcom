import json
import base64
import re
import os
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone

import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from mistralai import Mistral
from vosk import Model, KaldiRecognizer

# =========================
# CONFIG
# =========================
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "0c1Wsis4mHf2B2xSnBxXCC6lhWUBSax7")
MISTRAL_MODEL   = "mistral-medium-latest"
EVAL_MODEL      = "mistral-medium-latest"

INWORLD_API_KEY = os.environ.get("INWORLD_API_KEY", "OTdHdE1Hb0VseVM3RXhMVlNLYVFDMGcwOEZJbVF0eUY6OGhndjNhR3JhT0JyUXJqUWZWVXZqeWlTSFJRMDZSR3RTcllVRm9BS2VYUGFrTE9RTnpOQ0xteGlicTBzZGV3MQ==")
INWORLD_TTS_URL = "https://api.inworld.ai/tts/v1/voice:stream"

DEFAULT_VOICE_ID = "Hélène"
MODEL_ID  = "inworld-tts-1.5-max"

# Central place to pick a voice for a new module — add to this dict once,
# then reference VOICES["key"] in a module's "voice_id" instead of typing
# a raw Inworld voiceId (and risking a typo / wrong-gender voice).
VOICES = {
    "female_fr": "Hélène",
    "male_fr":   "Étienne",
    "male_fr_older": "Alain",
}

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-fr-0.22")

# =========================
# INTERVIEW MODULES
# The patient personas ("system") and evaluation prompts ("eval") live in modules.txt,
# next to this file, and are loaded once at startup by load_interview_modules() (see INIT).
# Each module has: label, practitioner_label, patient_label, voice_id, system, eval.
#   - "practitioner_label": how the human participant is labeled in the transcript sent to
#     the evaluator (and in the downloadable transcript), so evaluation feedback doesn't
#     default to "Médecin" for scenarios where the role is a generic healthcare provider.
# To add a module: add a block to modules.txt (format explained at the top of that file)
# and a matching entry in PATIENT_FICHES in index.html (same key).
# =========================

# =========================
# INIT
# =========================
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODULES_FILE = os.path.join(_BASE_DIR, "modules.txt")
_MODULE_RE = re.compile(
    r"^=== MODULE (\S+) ===\n(.*?)\n=== SYSTEM ===\n(.*?)\n=== EVAL ===\n(.*?)\n=== END MODULE ===$",
    re.M | re.S,
)

_MARKER_RE = re.compile(r"^=== (MODULE |SYSTEM ===|EVAL ===|END MODULE ===)", re.M)

def load_interview_modules(path: str) -> dict:
    """Read modules.txt once at startup and return the INTERVIEW_MODULES dict.
    Fails loudly (RuntimeError) on a malformed file, so a typo never goes live silently."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().replace("\r\n", "\n")      # tolerate files saved on Windows
    except OSError as e:
        raise RuntimeError(f"modules.txt introuvable ou illisible ({path}): {e}")

    modules = {}
    for m in _MODULE_RE.finditer(text):
        key, header, system, evaluation = m.groups()
        # a marker line inside a block means a neighbouring "=== END MODULE ===" is missing
        if _MARKER_RE.search("\n".join((header, system, evaluation))):
            raise RuntimeError(f"modules.txt: marqueur « === ... === » à l'intérieur du module '{key}' (« === END MODULE === » manquant ou mal placé ?)")
        if key in modules:
            raise RuntimeError(f"modules.txt: module '{key}' défini deux fois")
        fields = {}
        for line in header.split("\n"):
            if line.strip():
                name, sep, value = line.partition(":")
                if not sep:
                    raise RuntimeError(f"modules.txt: ligne d'en-tête invalide dans '{key}': {line!r}")
                fields[name.strip()] = value.strip()
        missing = [f for f in ("label", "practitioner_label", "patient_label", "voice") if not fields.get(f)]
        if missing:
            raise RuntimeError(f"modules.txt: module '{key}': champ(s) manquant(s): {', '.join(missing)}")
        if fields["voice"] not in VOICES:
            raise RuntimeError(f"modules.txt: module '{key}': voix inconnue '{fields['voice']}' (valides: {', '.join(VOICES)})")
        if not system.strip() or not evaluation.strip():
            raise RuntimeError(f"modules.txt: module '{key}': prompt SYSTEM ou EVAL vide")
        modules[key] = {
            "label": fields["label"],
            "system": system,
            "eval": evaluation,
            "practitioner_label": fields["practitioner_label"],
            "patient_label": fields["patient_label"],
            "voice_id": VOICES[fields["voice"]],
        }

    # anything outside the module blocks must be blank or a "#" comment line
    leftover = _MODULE_RE.sub("", text)
    stray = [l for l in leftover.split("\n") if l.strip() and not l.lstrip().startswith("#")]
    if stray:
        raise RuntimeError(f"modules.txt: texte en dehors d'un module (marqueurs mal formés ?): {stray[0][:80]!r}")
    if not modules:
        raise RuntimeError("modules.txt: aucun module trouvé")
    return modules

INTERVIEW_MODULES = load_interview_modules(MODULES_FILE)
print(f"[MODULES] {len(INTERVIEW_MODULES)} modules chargés depuis modules.txt: {', '.join(INTERVIEW_MODULES)}", flush=True)
app = Flask(
    __name__,
    template_folder=os.path.join(_BASE_DIR, "templates"),
    static_folder=os.path.join(_BASE_DIR, "static")
)


ALLOWED_ORIGINS = {
    "https://ulb-clientdev.edunao.com",
    "https://valerie-chatcom-uujl.onrender.com",
}


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        origin = request.headers.get("Origin", "")
        resp = app.make_default_options_response()
        if origin in ALLOWED_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp


@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin", "")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Vary"] = "Origin"
    return response

mistral_client = Mistral(api_key=MISTRAL_API_KEY)   # chat model
eval_client    = Mistral(api_key=MISTRAL_API_KEY)   # evaluation model

vosk_model     = Model(os.path.join(_BASE_DIR, VOSK_MODEL_PATH))

import sqlite3

# =========================
# PERSISTENT SESSION STORE
# A plain in-memory dict is wiped whenever the worker process restarts
# (e.g. a gunicorn WORKER TIMEOUT killing a hung request), losing every
# active student's conversation at once. SQLite on local disk survives
# a worker restart (same container, same filesystem) while still
# resetting on a genuine redeploy — which is fine, since a redeploy
# invalidates in-progress sessions anyway.
# =========================
SESSIONS_DB_PATH = os.path.join(_BASE_DIR, "sessions.db")

def _db_connect():
    conn = sqlite3.connect(SESSIONS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _db_init():
    with _db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                interview_type TEXT,
                history TEXT NOT NULL
            )
        """)

_db_init()

class _SessionStore:
    """Drop-in replacement for the old `sessions` dict, backed by SQLite.
    Supports the same `sessions[id]`, `sessions[id] = {...}`, `id in sessions`,
    `del sessions[id]` usage as before, so the routes below barely change."""

    def __contains__(self, session_id):
        with _db_connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return row is not None

    def __getitem__(self, session_id):
        with _db_connect() as conn:
            row = conn.execute(
                "SELECT interview_type, history FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        interview_type, history_json = row
        return {"interview_type": interview_type, "history": json.loads(history_json)}

    def __setitem__(self, session_id, value):
        with _db_connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, interview_type, history) VALUES (?, ?, ?)",
                (session_id, value.get("interview_type"), json.dumps(value["history"]))
            )

    def __delitem__(self, session_id):
        with _db_connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def save_history(self, session_id, history):
        """Persist an updated history list for an existing session without
        needing a full read-modify-write of the dict-like value."""
        with _db_connect() as conn:
            conn.execute("UPDATE sessions SET history = ? WHERE session_id = ?",
                         (json.dumps(history), session_id))

sessions = _SessionStore()

inworld_session = requests.Session()
inworld_session.headers.update({
    "Authorization": f"Basic {INWORLD_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "keep-alive"
})


# =========================
# HELPERS
# =========================

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


# Audio formats we can hand to the browser. Ogg/Opus is the default (best quality, unchanged
# behaviour for desktop browsers). iPhones/iPads cannot reliably play Ogg/Opus, so the page asks
# for MP3 there, which every browser plays.
AUDIO_FORMATS = {
    "ogg_opus": {"encoding": "OGG_OPUS", "sample_rate": 48000, "mimetype": "audio/ogg"},
    "mp3":      {"encoding": "MP3",      "sample_rate": 24000, "mimetype": "audio/mpeg"},
}

def normalize_audio_format(value) -> str:
    value = (value or "").strip().lower() if isinstance(value, str) else ""
    return value if value in AUDIO_FORMATS else "ogg_opus"

def stream_opus_chunks(text: str, voice_id: str = DEFAULT_VOICE_ID, audio_format: str = "ogg_opus"):
    fmt = AUDIO_FORMATS[normalize_audio_format(audio_format)]
    payload = {
        "text": text,
        "voiceId": voice_id,
        "modelId": MODEL_ID,
        "temperature": 1.48,
        "audio_config": {
            "audio_encoding": fmt["encoding"],
            "sample_rate_hz": fmt["sample_rate"],
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

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
                capture_output=True, check=True
            )
        except subprocess.CalledProcessError as e:
            # ffmpeg's actual reason (e.g. empty/corrupt input, unsupported
            # codec) is in stderr but subprocess.CalledProcessError's default
            # str() discards it, leaving only the unhelpful "exit status 1".
            stderr_text = (e.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg failed (exit {e.returncode}), input size={len(audio_bytes)} bytes: "
                f"{stderr_text[-1000:] if stderr_text else '(no stderr captured)'}"
            ) from e

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


@app.route("/modules", methods=["GET"])
def get_modules():
    return jsonify([
        {"key": k, "label": v["label"]}
        for k, v in INTERVIEW_MODULES.items()
    ])


@app.route("/start_session", methods=["POST"])
def start_session():
    data           = request.get_json(silent=True) or {}
    session_id     = (data.get("session_id") or "").strip()
    interview_type = (data.get("interview_type") or "").strip()

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if interview_type not in INTERVIEW_MODULES:
        print(f"[START_SESSION ERROR] invalid interview_type={interview_type!r} "
              f"ua={request.headers.get('User-Agent', '')[:120]}", flush=True)
        return jsonify({"error": f"Unknown interview_type '{interview_type}'",
                        "valid": list(INTERVIEW_MODULES.keys())}), 400

    system_prompt = INTERVIEW_MODULES[interview_type]["system"]
    sessions[session_id] = {
        "history": [{"role": "system", "content": system_prompt}],
        "interview_type": interview_type,
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
        tb = traceback.format_exc()
        app.config["LAST_ERROR"] = {"error": str(e), "trace": tb, "timestamp": datetime.now(timezone.utc).isoformat()}
        print(f"[TRANSCRIBE ERROR] audio={len(audio_bytes)} bytes -> {type(e).__name__}: {str(e)[-400:]}", flush=True)
        return jsonify({"error": str(e), "trace": tb}), 500


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()
    user_text  = data.get("message", "").strip()
    audio_format = normalize_audio_format(data.get("audio_format"))   # optional; default ogg_opus

    if not session_id or session_id not in sessions:
        return jsonify({"error": "Session not found — call /start_session first"}), 400
    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    session_row    = sessions[session_id]
    history        = session_row["history"]
    interview_type = session_row.get("interview_type")
    module         = INTERVIEW_MODULES.get(interview_type, {})
    voice_id       = module.get("voice_id", DEFAULT_VOICE_ID)

    history.append({"role": "user", "content": user_text})

    try:
        response = mistral_client.chat.complete(
            model=MISTRAL_MODEL,
            messages=history,
        )
        assistant_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_reply})
        # Persist the updated conversation — sessions[id] returns a fresh copy
        # each time (unlike the old in-memory dict), so mutating `history`
        # locally does nothing until it's explicitly written back here.
        sessions.save_history(session_id, history)
    except Exception as e:
        # `history` here is still just the local copy — nothing was saved
        # yet, so the stored session is untouched and this pop is only
        # cleaning up the local variable for correctness/clarity.
        history.pop()  # remove the orphaned user turn so history stays valid for the next attempt
        tb = traceback.format_exc()
        app.config["LAST_ERROR"] = {"error": str(e), "trace": tb, "timestamp": datetime.now(timezone.utc).isoformat()}
        print(f"[CHAT ERROR] session={session_id[:8]} -> {type(e).__name__}: {e}", flush=True)
        return jsonify({"error": str(e), "trace": tb}), 500

    sentences = split_sentences(assistant_reply)

    @stream_with_context
    def generate():
        for sentence in sentences:
            yield json.dumps({"type": "sentence_start", "text": sentence}) + "\n"
            try:
                for opus_chunk in stream_opus_chunks(sentence, voice_id, audio_format):
                    yield json.dumps({
                        "type": "audio",
                        "data": base64.b64encode(opus_chunk).decode()
                    }) + "\n"
            except Exception as e:
                tb = traceback.format_exc()
                app.config["LAST_ERROR"] = {"error": str(e), "trace": tb, "timestamp": datetime.now(timezone.utc).isoformat()}
                # Persistent trace in the Render logs: /debug_last_error only keeps the very last error.
                print(f"[TTS ERROR] session={session_id[:8]} voice={voice_id} chars={len(sentence)} -> {type(e).__name__}: {e}", flush=True)
            yield json.dumps({"type": "sentence_end"}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


@app.route("/end_session", methods=["POST"])
def end_session():
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400

    history        = sessions[session_id]["history"]
    interview_type = sessions[session_id].get("interview_type", "unknown")
    module         = INTERVIEW_MODULES.get(interview_type, {})
    label          = module.get("label", interview_type)
    practitioner_label = module.get("practitioner_label", "Soignant")
    patient_label       = module.get("patient_label", "Patiente")

    lines = [
        "=== Transcript de consultation ===",
        f"Module      : {label}",
        f"Session     : {session_id}",
        "",
    ]
    for msg in history:
        if msg["role"] == "system":
            continue
        speaker = f"{practitioner_label}  " if msg["role"] == "user" else f"{patient_label} "
        lines.append(f"{speaker}: {msg['content']}")
        lines.append("")

    transcript = "\n".join(lines)
    return Response(
        transcript,
        mimetype="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename=transcript_{session_id[:8]}.txt"
        }
    )


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 400

    session_data   = sessions[session_id]
    history        = session_data["history"]
    interview_type = session_data.get("interview_type")

    module = INTERVIEW_MODULES.get(interview_type)
    if not module or "eval" not in module:
        return jsonify({
            "error": f"No evaluation prompt configured for interview_type '{interview_type}'"
        }), 400

    eval_system_prompt = module["eval"]
    practitioner_label  = module.get("practitioner_label", "Soignant")
    patient_label       = module.get("patient_label", "Patiente")

    lines = []
    for msg in history:
        if msg["role"] == "system":
            continue
        speaker = practitioner_label if msg["role"] == "user" else patient_label
        lines.append(f"{speaker}: {msg['content']}")
    transcript_text = "\n".join(lines)

    if not transcript_text.strip():
        return jsonify({"error": "Transcript is empty"}), 400

    try:
        response = eval_client.chat.complete(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": eval_system_prompt},
                {"role": "user",   "content": f"Voici le transcript de la consultation à évaluer :\n\n{transcript_text}"}
            ]
        )
        feedback = response.choices[0].message.content.strip()
        return jsonify({"feedback": feedback})
    except Exception as e:
        tb = traceback.format_exc()
        app.config["LAST_ERROR"] = {"error": str(e), "trace": tb, "timestamp": datetime.now(timezone.utc).isoformat()}
        return jsonify({"error": str(e), "trace": tb}), 500


@app.route("/debug_last_error", methods=["GET"])
def debug_last_error():
    return jsonify(app.config.get("LAST_ERROR", "no error recorded yet"))


@app.route("/history", methods=["GET"])
def get_history():
    session_id = request.args.get("session_id", "")
    if session_id not in sessions:
        return jsonify([])
    return jsonify([m for m in sessions[session_id]["history"] if m["role"] != "system"])


@app.route("/reset", methods=["POST"])
def reset_history():
    data       = request.get_json()
    session_id = data.get("session_id", "").strip()
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "reset"})


# =========================
# AUDIO SELF-TEST (diagnostic page for devices where the patient's voice is silent)
# =========================
AUDIO_TEST_PHRASE = "Bonjour, ceci est un test audio. Si vous m'entendez, tout fonctionne."
_audio_test_clips = {}   # one cached clip per format: at most 2 TTS calls per server start

AUDIO_TEST_HTML = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test audio — ChatCom</title>
<style>
  body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:0;padding:18px;max-width:560px;margin:auto;color:#222;background:#f5f0e8;line-height:1.45}
  h1{font-size:1.3rem;margin:.2em 0 .4em} h2{font-size:1rem;margin:1.4em 0 .4em}
  p{margin:.4em 0} .box{background:#fff;border:1px solid #d9cfbf;border-radius:10px;padding:12px;margin:10px 0}
  button{font-size:1rem;padding:10px 14px;border-radius:8px;border:1px solid #8b5e3c;background:#8b5e3c;color:#fff;margin:4px 6px 4px 0}
  button.ans{background:#fff;color:#8b5e3c;padding:7px 12px} .small{font-size:.85rem;color:#666}
  pre{background:#222;color:#e8e8e8;padding:12px;border-radius:8px;white-space:pre-wrap;word-break:break-word;font-size:.78rem}
</style></head><body>
<h1>Test audio ChatCom</h1>
<p>1. Montez le volume et désactivez le mode silencieux de l'appareil (interrupteur sur le côté).<br>
2. Appuyez sur chaque bouton, puis répondez « Oui » ou « Non » : avez-vous entendu la phrase ?<br>
3. À la fin, faites une capture d'écran du résultat en bas de page.</p>

<div class="box"><b>Test 1</b> — MP3, lecture classique<br>
 <button data-test="1">▶ Lancer</button>
 <span class="small">Entendu ?</span> <button class="ans" data-ans="1:oui">Oui</button><button class="ans" data-ans="1:non">Non</button></div>

<div class="box"><b>Test 2</b> — Ogg/Opus (ancien format), lecture classique<br>
 <button data-test="2">▶ Lancer</button>
 <span class="small">Entendu ?</span> <button class="ans" data-ans="2:oui">Oui</button><button class="ans" data-ans="2:non">Non</button></div>

<div class="box"><b>Test 3</b> — MP3, lecteur « déverrouillé » (nouvelle méthode)<br>
 <button data-test="3">▶ Lancer</button>
 <span class="small">Entendu ?</span> <button class="ans" data-ans="3:oui">Oui</button><button class="ans" data-ans="3:non">Non</button></div>

<div class="box"><b>Test 4</b> — MP3, lecteur déverrouillé <u>avec le micro ouvert</u> (autorisez le micro si demandé)<br>
 <button data-test="4">▶ Lancer</button>
 <span class="small">Entendu ?</span> <button class="ans" data-ans="4:oui">Oui</button><button class="ans" data-ans="4:non">Non</button></div>

<h2>Résultat (à capturer)</h2>
<pre id="log"></pre>

<script>
const out = document.getElementById("log");
const log = (m) => { out.textContent += m + "\n"; };
const SILENT = "data:audio/wav;base64,UklGRsQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YaAAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICA";
const shared = new Audio();            // un seul lecteur, déverrouillé par un appui, réutilisé ensuite
let micStream = null;

function cap(){
  const a = document.createElement("audio");
  const s = (t) => a.canPlayType(t) || "non";
  return { ogg: s('audio/ogg; codecs="opus"'), mp3: s("audio/mpeg") };
}
const c = cap();
log("Date : " + new Date().toISOString());
log("Appareil : " + navigator.userAgent);
log("Lecture Ogg/Opus : " + c.ogg + "  |  MP3 : " + c.mp3);
log("audioSession : " + (navigator.audioSession ? ("oui, type=" + navigator.audioSession.type) : "absent"));
log("");

async function run(n){
  const fmt = (n === 2) ? "ogg_opus" : "mp3";
  const unlocked = (n === 3 || n === 4);
  const mic = (n === 4);
  log("--- Test " + n + " (" + fmt + (unlocked ? ", lecteur déverrouillé" : ", lecteur classique") + (mic ? ", micro ouvert" : "") + ") ---");
  let player = null;
  if (unlocked) {                      // doit se faire tout de suite, dans l'appui
    try { shared.src = SILENT; const p = shared.play(); if (p && p.catch) p.catch(()=>{}); } catch(e){}
  }
  try {
    if (mic) { micStream = await navigator.mediaDevices.getUserMedia({audio:true}); log("micro : ouvert"); }
    const r = await fetch("/audiotest/clip?format=" + fmt);
    if (!r.ok) { log("serveur : erreur " + r.status + " " + (await r.text()).slice(0,200)); return; }
    const blob = await r.blob();
    log("clip reçu : " + blob.size + " octets, type=" + blob.type);
    const url = URL.createObjectURL(blob);
    player = unlocked ? shared : new Audio();
    player.onended = () => { log("lecture terminée"); URL.revokeObjectURL(url); if (micStream) { micStream.getTracks().forEach(t=>t.stop()); micStream = null; } };
    player.onerror = () => log("ERREUR média : code=" + (player.error && player.error.code) + " (4 = format non pris en charge)");
    player.src = url;
    await player.play();
    log("lecture démarrée");
  } catch (e) { log("ERREUR : " + e.name + " — " + e.message); }
}

document.querySelectorAll("button[data-test]").forEach(b => b.addEventListener("click", () => run(Number(b.dataset.test))));
document.querySelectorAll("button[data-ans]").forEach(b => b.addEventListener("click", () => {
  const [n, a] = b.dataset.ans.split(":");
  log("=> Test " + n + " : " + (a === "oui" ? "ENTENDU" : "PAS entendu"));
}));
</script>
</body></html>
"""


@app.route("/audiotest", methods=["GET"])
def audiotest_page():
    return Response(AUDIO_TEST_HTML, mimetype="text/html")


@app.route("/audiotest/clip", methods=["GET"])
def audiotest_clip():
    fmt_key = normalize_audio_format(request.args.get("format"))
    if fmt_key not in _audio_test_clips:
        try:
            _audio_test_clips[fmt_key] = b"".join(
                stream_opus_chunks(AUDIO_TEST_PHRASE, VOICES["female_fr"], fmt_key)
            )
        except Exception as e:
            tb = traceback.format_exc()
            app.config["LAST_ERROR"] = {"error": str(e), "trace": tb, "timestamp": datetime.now(timezone.utc).isoformat()}
            print(f"[AUDIOTEST ERROR] format={fmt_key} -> {type(e).__name__}: {e}", flush=True)
            return jsonify({"error": str(e)}), 500
    clip = _audio_test_clips[fmt_key]
    if not clip:
        return jsonify({"error": "TTS returned no audio"}), 502
    return Response(clip, mimetype=AUDIO_FORMATS[fmt_key]["mimetype"], headers={"Cache-Control": "no-store"})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
