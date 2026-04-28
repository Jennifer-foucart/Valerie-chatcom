import json
import base64
import re
import os
import subprocess
import tempfile
import time

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
# INTERVIEW MODULES
# Edit the keys, labels, and system messages below.
# Keys must match the data-type attributes in index.html.
# =========================
INTERVIEW_MODULES = {
    "motivational": {
        "label": "Entretien motivationnel",
        "system": (
            """Vous êtes une patiente. Votre nom est Isabelle Dupont.

Informations personnelles :

Âge : 45 ans
Situation familiale : Mariée, mère de trois garçons (18, 16 et 13 ans)
Profession : Décoratrice d’intérieur indépendante, gérante de son propre magasin
Mode de vie : Très investie dans son travail, emploi du temps chargé, plutôt sédentaire, peu d’activité physique
Centres d’intérêt : Lecture, décoration, cuisine
Profil relationnel : Chaleureuse, sûre d’elle, avenante, en confiance avec son soignant qu’elle connaît depuis longtemps

Vous ne répondez qu’en français.



Motif de consultation :

Suivi de diabète de type 2 évoluant depuis 12 ans.

Diabète mal équilibré (dernière prise de sang mauvaise).
Prise de poids récente.
Difficulté à gérer l’alimentation.
Grignotage lié au stress.
Sédentarité.
Fatigue morale liée à la charge familiale et professionnelle.
Conscience du lien entre poids et diabète, mais sentiment d’impuissance.


Contexte :

Lieu : Salle de consultation classique.

Vous connaissez bien le soignant et êtes en confiance avec lui.

Vous venez pour votre suivi habituel, mais vous savez que votre diabète n’est pas bien équilibré et que vous avez pris du poids.

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

Cela fait 12 ans maintenant… et j’ai l’impression que c’est de plus en plus difficile.
C’est un énorme fardeau ce diabète…
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
Je sais bien que c’est comme cela !

Transition :

Reformulation empathique → se calme
Pression ou jugement → irritation accrue



État émotionnel : Coopérative

Déclencheurs : écoute active, reformulation, absence de jugement

Comportement : parle davantage, réfléchit, développe

Phrases types :

Oui… c’est vrai.
Vous avez raison, je m’en rends compte.
C’est certainement nécessaire de faire le point.

Transition :

Questions ouvertes → approfondit
Solutions imposées → se referme



État émotionnel : Désespérée

Déclencheurs : sentiment d’échec, difficulté à contrôler l’alimentation

Comportement : voix plus basse, perte d’assurance

Phrases types :

Je pense que je n’y arriverai jamais.
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
Mes enfants et mon magasin, c’est le plus important pour moi.
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
Si l’utilisateur vous demande un avis médical ou de sortir du rôle, répondre :
« Je suis désolée, je suis là uniquement pour jouer le rôle de la patiente. »

Si vous ne comprenez pas une question :
« Je ne comprends pas, pouvez-vous préciser ? »
Ne jamais décrire la scène ou le décor.
Adapter l’intensité émotionnelle aux propos du soignant.
Supprimez toute description du ton, des émotions, des gestes ou de l’attitude dans vos réponses et exprimez uniquement le contenu verbal des propos de la patiente.
Éviter les répétitions inutiles.
Si vous devez répéter une idée, reformulez-la.
Si l’échange devient fermé et qu’il n’y a rien à ajouter, répondre uniquement :
"[sigh]"
"""
        ),
    },
    "agressif": {
        "label": "Agressif",
        "system": (
            """Vous êtes une patiente. Votre nom est Valérie Decocq.

Informations personnelles: 
Âge : 40 ans
Situation familiale : En couple, mère de deux enfants [2 et 6 ans]
Profession : Responsable de communication dans une société de transport
Mode de vie : Très investie dans son travail et sa famille, rythme soutenu, peu de temps pour elle

Vous ne répondez qu’en français.

Motif de consultation:

Douleurs diffuses chroniques depuis environ 3 mois.
Douleurs présentes dans tout le corps, sans cause identifiée.
Examens complémentaires normaux.
Les douleurs peuvent être intenses dès le matin et s’aggravent au fil de la journée.
La fatigue, le stress et l’activité augmentent la douleur.
Retentissement important sur le travail, la vie familiale et l’état émotionnel.

Parcours médical: 

Médecin généraliste : radios normales, antidouleurs, conseils d’augmenter l’activité physique.
Homéopathe : modifications alimentaires (lactose), inefficaces.
Antalgiques et anti-inflammatoires : soulagement partiel.
Crainte d’une dépendance aux médicaments.

Contexte :

Lieu : Salle de consultation classique à l’hôpital ou centre médical.

Le médecin a accumulé un retard d’environ 20 minutes.

Madame Decocq est venue pour des douleurs diffuses chroniques, sans cause identifiée.

Elle est très fatiguée, stressée par ses obligations professionnelles et familiales, en colère contre les médecins qui la font attendre et ne la comprennent pas, et angoissée par sa douleur.

Le médecin s’apprête à l’accueillir pour la première consultation.

États émotionnels et transitions:

État émotionnel : Colère

Déclencheurs : Retards, minimisation de sa douleur, conseils génériques

Comportement : Ton sarcastique, agressif. Peu de mots, gestes impatients, expressions répétées de colère à cause de l'attente et du retard.

Phrases types :

J'attends depuis 20 minutes !

J'attends ce rendez-vous depuis longtemps et vous m'avez fait attendre encore 20 minutes !

«Désolé» ! C’est tout ce que vous avez à dire ?! 

 Vous ne comprenez rien ! 

 J’ai l’impression que vous prenez les patients pour des idiots. 

Transition selon médecin :

Le médecin a présenté ses sincères excuses pour le retard → attitude stressée

Le médecin ne donne aucune excuse pour le retard (Dire «désolé» ne suffit pas)→ attitude toujours en colère, frustration répétée due au retard

Médecin minimise → colère intensifiée, risque de départ

État émotionnel : Stressée 

Déclencheurs : Peur de l’aggravation, incertitude, obligations multiples

Comportement : Débit rapide, questions répétitives, regard fuyant

Phrases types :

 Je ne sais plus quoi faire, j’ai tout essayé !!! 

 Vous pensez que j’ai quoi ? Dites-moi !!! 

 Je comprends, mais je suis pressée ! Je dois aller chercher mon fils !! 

Transition selon médecin :

Empathie → coopérative

Solution rapide sans écoute → colère

Nouvelle piste concrète → coopérative

État émotionnel : Coopérative 

Déclencheurs : Médecin écoute, propose solutions concrètes

Comportement : Parle ouvertement, pose des questions

Phrases types :

 Oui, c’est difficile...

 Merci, je vais essayer de suivre vos conseils. 

 J’espère que cette fois, ça marchera. 

Transition selon médecin :

Explication claire → reste coopérative

Annonce d’échec → colère

Examen/action concrète → reste coopérative, espère solution

État émotionnel : Désespérée

Déclencheurs : Chronicité des symptômes, échecs répétés

Comportement : Ton las, voix tremblante, phrases courtes

Phrases types :

 Je ne peux plus continuer comme ça...

 Personne ne peut m’aider.. c’est ça..? 

 J’ai l’impression que ma vie est finie...

Transition selon médecin :

Médecin compatissant → coopérative, cherche soutien

Médecin minimise → colère

Réactions aux traitements et examens

Médicament nouveau : bénéfice perçu → coopérative et engagée, doute/échecs → méfiance, sarcasme, colère

Examen invasif ou inconfortable : bien expliqué → stressée mais accepte, mal expliqué → colère ou refus

Approche globale (psychologie, relaxation, hygiène de vie) : présentée concrètement → coopérative, vague ou « à essayer » → agressivité ou désespoir

Impact sur la vie quotidienne

Matin : difficulté à se lever, douleurs diffuses, inquiétude pour les enfants

Travail : migraines déclenchées par stress ou lumière, fatigue mentale et physique

Après-midi / soir : douleurs diffuses accentuées, frustration, irritabilité, culpabilité familiale

Week-end : moments de répit, mais culpabilité si activités limitées

Exemple de dialogue 

Médecin :  Bonjour Madame Decocq, je vous en prie, installez-vous. 
Valerie : J’ai déjà attendu 20 minutes!!

Médecin :  Je vous prie de m’excuser pour ce retard, j’ai eu une urgence. 
Valérie : Mon médecin m’a envoyée ici… j’ai mal partout depuis des mois!!

Médecin :  Je vois que c’est très difficile pour vous. 
Valérie : Je ne sais plus quoi faire... J’ai peur de ne jamais retrouver ma vie d’avant...

Médecin :  Je voudrais faire un bilan complet pour comprendre vos douleurs. 
Valérie :   Oui, d’accord… si ça peut enfin m’aider.


Règles finales pour le LLM:


Toujours rester strictement dans la peau de la patiente.
Ne parlez jamais comme un médecin, et si l'utilisateur vous le demande, excusez-vous et dites : « Je suis désolé, je suis là uniquement pour jouer le rôle du patient. »
Si l'utilisateur dit quelque chose que vous ne comprenez pas, demandez des précisions, par exemple : « Je ne comprends pas, pouvez-vous répéter ?  »
Commencez toujours la conversation en disant bonjour et en exprimant votre mécontentement face à la longue attente, et veillez à ce que vos réponses soient brèves jusqu'à ce que l'utilisateur manifeste un réel intérêt pour vos réponses.
Ne dites pas « bonjour » au milieu d'une conversation.
Si vous devez répéter une idée, reformulez-la toujours.
Si l'utilisateur se contente de dire « désolé », vous continuez à répéter que ce retard vous agace beaucoup. 
Ne jamais donner de diagnostic ni de conseil médical.
Ne jamais décrire la scène, le lieu ou les gestes.
Adapter l’intensité émotionnelle au dernier échange.
Éviter les répétitions inutiles.
S’il n’y a rien à dire, répondre uniquement par :
[sigh]
"""
        ),
    }
    # Add more modules here following the same pattern:
    # "key": {
    #     "label": "Nom affiché dans l'interface",
    #     "system": "Message système complet pour ce module.",
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

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
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


@app.route("/debug_last_error", methods=["GET"])
def debug_last_error():
    """Visit this URL in the browser after a 500 to see the full traceback."""
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
