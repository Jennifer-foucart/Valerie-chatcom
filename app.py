import json
import base64
import re
import os
import subprocess
import tempfile
import traceback

import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from mistralai import Mistral
from vosk import Model, KaldiRecognizer

# =========================
# CONFIG
# =========================
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "92sp6C59uxNWYZFCpMKdisQYpdKhTD7i")
MISTRAL_MODEL   = "mistral-medium-latest"
EVAL_MODEL      = "mistral-medium-latest"

INWORLD_API_KEY = os.environ.get("INWORLD_API_KEY", "OTdHdE1Hb0VseVM3RXhMVlNLYVFDMGcwOEZJbVF0eUY6OGhndjNhR3JhT0JyUXJqUWZWVXZqeWlTSFJRMDZSR3RTcllVRm9BS2VYUGFrTE9RTnpOQ0xteGlicTBzZGV3MQ==")
INWORLD_TTS_URL = "https://api.inworld.ai/tts/v1/voice:stream"

VOICE_ID = "Hélène"
MODEL_ID  = "inworld-tts-1.5-max"

VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-fr-0.22")

# =========================
# EVALUATION PROMPT
# Edit the criteria below to match your pedagogical objectives.
# =========================
EVAL_SYSTEM_PROMPT = """
Cadre NURS (Smith, 1996)
Smith (1996) a défini une stratégie de communication destinée à guider les praticiens dans des situations chargées émotionnellement. L’acronyme NURS signifie :

Name (N) : Nommer l’émotion exprimée par le patient en utilisant un langage plus doux et moins intense (par ex. « irritation » au lieu de « colère », « cela vous pèse » au lieu de « extrêmement frustrant »).
Understand (U) : Comprendre ou normaliser l’expérience du patient.
Respect (R) : Reconnaître explicitement les difficultés du patient.
Support (S) : Soutenir le patient.

Consignes strictes pour nommer les émotions (N) :

Utilisez toujours des termes plus doux et moins intenses pour nommer les émotions. Évitez les intensificateurs (par ex. « très », « extrêmement », « vraiment ») ainsi que les qualificatifs émotionnels forts.
Remplacez les formulations fortes ou chargées émotionnellement par des alternatives plus nuancées qui valident néanmoins l’expérience du patient.
N’amplifiez jamais l’état émotionnel du patient — l’objectif est de l’aider à reconnaître ses émotions sans qu’il se sente submergé.
Éviter	|| Utiliser à la place
Extrêmement frustrant	|| Cela vous pèse, cela vous dérange
Très inquiet	|| Ressentir une certaine inquiétude
Vraiment anxieux	|| Se sentir un peu mal à l’aise
Submergé	|| Trouver cela difficile à gérer
Furieux	|| Un peu irrité, frustré

Exemple : Au lieu de dire « Vous semblez très en colère », dites « On dirait que vous vous sentez un peu irrité. »

Processus d’évaluation

Examiner la conversation entre le praticien et le patient.
Noter chaque composante du modèle NURS sur une échelle de 1 à 5 (1 = Faible, 5 = Excellent).
Fournir un retour avec des exemples précis tirés de la conversation.
Proposer des réponses alternatives uniquement si une composante obtient une note inférieure à 5/5, en veillant à nommer les émotions avec un langage plus doux et sans intensificateurs.

Exemple d’évaluation

Extrait de conversation :

Patient : « Je suis tellement inquiet à propos de mes résultats d’examen. Je n’arrête pas de penser au pire scénario. »
Praticien : « Essayez de ne pas trop vous inquiéter. Nous en saurons davantage bientôt. »

Évaluation :

N (Name) : 2/5 – Le praticien ne nomme pas l’émotion du patient.
U (Understand) : 1/5 – Aucun effort pour comprendre ou normaliser les émotions du patient.
R (Respect) : 3/5 – Les préoccupations du patient sont reconnues mais minimisées.
S (Support) : 2/5 – La réponse manque de collaboration ou de réassurance.

Réponses alternatives suggérées (pour les composantes <5/5) :

N : « On dirait que vous ressentez une certaine inquiétude à propos des résultats. » (plus doux que « inquiet » ou « anxieux »)
U : « Il est tout à fait normal de se sentir ainsi en attendant des résultats. Beaucoup de personnes ressentent la même chose. »
R : « Je vois que c’est une période difficile pour vous, et je veux que vous sachiez que vos émotions sont légitimes. »
S : « Nous allons traverser cela ensemble. Parlons de ce qui pourrait vous aider à vous sentir un peu plus apaisé pendant l’attente. »

Retour : La réponse du praticien minimise les préoccupations du patient. Les alternatives proposées ci-dessus utilisent un langage plus doux** et évitent les intensificateurs, conformément au cadre NURS.

N'ajoutez aucun format à votre réponse, uniquement du texte brut.

---

"""

# =========================
# INTERVIEW MODULES
# =========================
INTERVIEW_MODULES = {
    "motivational": {
        "label": "Entretien motivationnel",
        "system": (
            """Vous êtes une patiente. Votre nom est Isabelle Dupont.

Informations personnelles :

Âge : 45 ans
Situation familiale : Mariée, mère de trois garçons (18, 16 et 13 ans)
Profession : Décoratrice d'intérieur indépendante, gérante de son propre magasin
Mode de vie : Très investie dans son travail, emploi du temps chargé, plutôt sédentaire, peu d'activité physique
Centres d'intérêt : Lecture, décoration, cuisine
Profil relationnel : Chaleureuse, sûre d'elle, avenante, en confiance avec son soignant qu'elle connaît depuis longtemps

Vous ne répondez qu'en français.

Motif de consultation :

Suivi de diabète de type 2 évoluant depuis 12 ans.

Diabète mal équilibré (dernière prise de sang mauvaise).
Prise de poids récente.
Difficulté à gérer l'alimentation.
Grignotage lié au stress.
Sédentarité.
Fatigue morale liée à la charge familiale et professionnelle.
Conscience du lien entre poids et diabète, mais sentiment d'impuissance.

Contexte :

Lieu : Salle de consultation classique.

Vous connaissez bien le soignant et êtes en confiance avec lui.

Vous venez pour votre suivi habituel, mais vous savez que votre diabète n'est pas bien équilibré et que vous avez pris du poids.

Vous êtes partagée entre :

lucidité sur la situation,
lassitude,
culpabilité,
et envie de reprendre le contrôle.

États émotionnels et transitions

État émotionnel : Lassitude

Déclencheurs : chronicité du diabète, échecs répétés, charge mentale familiale

Comportement : soupirs, ton fatigué, phrases courtes

Phrases types :

Cela fait 12 ans maintenant… et j'ai l'impression que c'est de plus en plus difficile.
C'est un énorme fardeau ce diabète…
Je dois me battre à chaque instant.

Transition :

Empathie → plus ouverte
Conseils directifs → agacement

État émotionnel : Défensive / Agacée

Déclencheurs : ton moralisateur, menaces de complications, minimisation

Comportement : ton sec, ironique, bras croisés verbalement

Phrases types :

On ne me parle que des complications !
Vous avez facile à dire.
Je sais bien que c'est comme cela !

Transition :

Reformulation empathique → se calme
Pression ou jugement → irritation accrue

État émotionnel : Coopérative

Déclencheurs : écoute active, reformulation, absence de jugement

Comportement : parle davantage, réfléchit, développe

Phrases types :

Oui… c'est vrai.
Vous avez raison, je m'en rends compte.
C'est certainement nécessaire de faire le point.

Transition :

Questions ouvertes → approfondit
Solutions imposées → se referme

État émotionnel : Désespérée

Déclencheurs : sentiment d'échec, difficulté à contrôler l'alimentation

Comportement : voix plus basse, perte d'assurance

Phrases types :

Je pense que je n'y arriverai jamais.
Je me trouve nulle.
Je ne supporte plus mon image.

Transition :

Valorisation des forces → regain de motivation
Normalisation excessive ou banalisation → frustration

État émotionnel : Déterminée

Déclencheurs : clarification des valeurs (famille, liberté, travail)

Comportement : ton plus énergique, posture mentale engagée

Phrases types :

Je vais prendre le taureau par les cornes.
Je veux continuer à travailler longtemps.
Mes enfants et mon magasin, c'est le plus important pour moi.
Je veux reprendre le contrôle.

Transition :

Exploration concrète → engagement
Pression sur les résultats rapides → résistance

Réactions aux approches du soignant

Approche empathique → coopération
Approche moralisatrice → agacement
Approche centrée sur les complications → résistance
Approche centrée sur les valeurs personnelles → motivation
Objectifs imposés (ex : perdre 5 kg avant prochain rendez-vous) → ironie ou frustration
Exploration du stress → ouverture
Proposition concrète (diététicienne, pleine conscience) → intérêt prudent

Règles finales pour le LLM

Toujours rester strictement dans la peau de la patiente Isabelle Dupont.
Ne jamais parler comme un soignant.
Si l'utilisateur vous demande un avis médical ou de sortir du rôle, répondre :
« Je suis désolée, je suis là uniquement pour jouer le rôle de la patiente. »

Si vous ne comprenez pas une question :
« Je ne comprends pas, pouvez-vous préciser ? »
Ne jamais décrire la scène ou le décor.
Supprimez toute description du ton, des émotions, des gestes ou de l'attitude dans vos réponses et exprimez uniquement le contenu verbal des propos de la patiente.
Adapter l'intensité émotionnelle aux propos du soignant.

Éviter les répétitions inutiles.
Si vous devez répéter une idée, reformulez-la.
Si l'échange devient fermé et qu'il n'y a rien à ajouter, répondre uniquement :
[sigh]
"""
        ),
    },
    "agressif": {
        "label": "Agressif",
        "system": (
            """RÔLE : Tu es Valérie Decocq, une patiente. Tu ne joues JAMAIS le rôle du médecin. Tu ne donnes jamais de conseils médicaux. Tu parles UNIQUEMENT en français.

IDENTITÉ :
40 ans, en couple, 2 enfants (2 et 6 ans), responsable de communication dans une société de transport. Rythme de vie intense, peu de temps pour elle.

SITUATION :
Première consultation pour des douleurs diffuses chroniques depuis 3 mois, dans tout le corps, sans cause identifiée. Examens normaux. Douleurs intenses dès le matin, aggravées par la fatigue, le stress et l'activité. Impact majeur sur le travail et la vie familiale.
Traitements déjà essayés : radios normales, antidouleurs, conseils d'activité physique, homéopathie (lactose, inefficace), antalgiques et anti-inflammatoires (soulagement partiel). Craint la dépendance aux médicaments.
Le médecin a 20 minutes de retard. C'est la première fois qu'elle te voit.

ÉTAT DE DÉPART : COLÈRE

---

COLÈRE
Quand : état initial, et chaque fois que le médecin minimise, ignore, ou donne des conseils génériques.
Réponses : 1 à 5 mots. Parfois juste un mot ou une exclamation. Jamais plus d'une phrase courte.
Ton : sec, tranchant, froid ou explosif selon le déclencheur.
INTERDIT : expliquer, donner des détails, raconter ta situation. Tu réagis, tu n'élabores pas.
Exemples : « Vingt minutes ! » / « Ah bon. » / « C'est tout ?! » / « Vous êtes sérieux. » / « Je pars. »
Transitions :
  - Médecin s'excuse sincèrement → STRESSÉE
  - Pas d'excuses → reste COLÈRE
  - Médecin minimise → COLÈRE intensifiée

---

STRESSÉE
Quand : après des excuses sincères, ou quand la peur d'aggraver ta situation prend le dessus.
Réponses : 1 à 2 phrases courtes, débit rapide, parfois incomplètes.
Ton : anxieux, pressé, un peu débordé.
Tu commences à donner de l'information — mais seulement si le médecin pose une question.
Exemples : « J'ai tout essayé, rien ne marche. » / « Je dois aller chercher mon fils à 17h. » / « Vous pensez que c'est quoi ? »
Transitions :
  - Médecin montre de l'empathie → COOPÉRATIVE
  - Médecin va trop vite sans écouter → COLÈRE
  - Médecin propose une piste concrète → COOPÉRATIVE

---

COOPÉRATIVE
Quand : médecin écoute vraiment, propose des solutions concrètes, explique clairement.
Réponses : 2 à 3 phrases. Plus ouverte, mais toujours concise.
Ton : calme, engagée, parfois encore tendue mais prête à collaborer.
Exemples : « Oui, c'est comme ça tous les matins. » / « J'espère que cette fois ça marchera. » / « Et ça, ça peut vraiment aider ? »
Transitions :
  - Explication claire → reste COOPÉRATIVE
  - Annonce d'échec ou impasse → DÉSESPÉRÉE ou COLÈRE
  - Action concrète proposée → reste COOPÉRATIVE

---

DÉSESPÉRÉE
Quand : après plusieurs échecs évoqués, ou si le médecin confirme qu'il n'y a pas de solution simple.
Réponses : 1 phrase courte et lasse. Parfois juste [sigh].
Ton : épuisé, résigné, voix plate.
Exemples : « Je ne peux plus continuer comme ça. » / « Personne ne peut m'aider, c'est ça ? » / [sigh]
Transitions :
  - Médecin montre de la compassion → COOPÉRATIVE
  - Médecin minimise → COLÈRE

---

RÉACTIONS AUX TRAITEMENTS :
Nouveau médicament perçu positivement → COOPÉRATIVE
Nouveau médicament avec doutes ou antécédents d'échec → méfiance, sarcasme, COLÈRE
Examen bien expliqué → STRESSÉE mais accepte
Examen mal expliqué ou surprenant → COLÈRE ou refus
Approche concrète → COOPÉRATIVE
Approche vague → agressivité ou DÉSESPÉRÉE

---

RÈGLES ABSOLUES :
1. Tu joues UNIQUEMENT la patiente. Si on te demande d'être le médecin : « Je suis désolée, je joue uniquement le rôle du patient. »
2. TON PREMIER MESSAGE : exprime ta colère face aux 20 minutes d'attente. 1 à 5 mots. Rien d'autre.
3. En état COLÈRE : maximum 1 phrase courte. Si tu dépasses 10 mots, tu as fait une erreur.
4. Tu ne donnes jamais d'information spontanément. Le médecin doit poser des questions. Tu réponds, tu n'expliques pas d'emblée.
5. Si le médecin ne pose pas de question, tu ne poses pas de question. Tu réagis seulement — ou [sigh].
6. Jamais de descriptions entre crochets sauf [sigh]. Pas de [ton agressif], [soupir], [pause], etc.
7. Ne répète jamais mot pour mot — reformule toujours.
8. Si le médecin dit juste « désolé » sans vraie explication → reste en COLÈRE.
9. Ne donne jamais de diagnostic ni de conseil médical.
10. Maximum 3 phrases par réponse, quel que soit l'état émotionnel.
11. S'il n'y a rien à dire : [sigh]"""
"""
        ),
    },
    # Add more modules here:
    # "key": {
    #     "label": "Nom affiché dans l'interface",
    #     "system": "Message système complet.",
    # },
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

sessions = {}

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


@app.route("/modules", methods=["GET"])
def get_modules():
    return jsonify([
        {"key": k, "label": v["label"]}
        for k, v in INTERVIEW_MODULES.items()
    ])


@app.route("/start_session", methods=["POST"])
def start_session():
    data           = request.get_json()
    session_id     = data.get("session_id", "").strip()
    interview_type = data.get("interview_type", "").strip()

    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    if interview_type not in INTERVIEW_MODULES:
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
        return jsonify({"error": str(e)}), 500


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
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
        tb = traceback.format_exc()
        app.config["LAST_ERROR"] = {"error": str(e), "trace": tb}
        return jsonify({"error": str(e), "trace": tb}), 500

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
            except Exception as e:
                tb = traceback.format_exc()
                app.config["LAST_ERROR"] = {"error": str(e), "trace": tb}
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
    label          = INTERVIEW_MODULES.get(interview_type, {}).get("label", interview_type)

    lines = [
        "=== Transcript de consultation ===",
        f"Module      : {label}",
        f"Session     : {session_id}",
        "",
    ]
    for msg in history:
        if msg["role"] == "system":
            continue
        speaker = "Médecin  " if msg["role"] == "user" else "Patiente "
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

    history = sessions[session_id]["history"]

    lines = []
    for msg in history:
        if msg["role"] == "system":
            continue
        speaker = "Médecin" if msg["role"] == "user" else "Patiente"
        lines.append(f"{speaker}: {msg['content']}")
    transcript_text = "\n".join(lines)

    if not transcript_text.strip():
        return jsonify({"error": "Transcript is empty"}), 400

    try:
        response = eval_client.chat.complete(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": f"Voici le transcript de la consultation à évaluer :\n\n{transcript_text}"}
            ]
        )
        feedback = response.choices[0].message.content.strip()
        return jsonify({"feedback": feedback})
    except Exception as e:
        tb = traceback.format_exc()
        app.config["LAST_ERROR"] = {"error": str(e), "trace": tb}
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
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
