# ChatCom — Documentation technique

*Document complémentaire au guide d'exploitation non technique. Celui-ci s'adresse à une personne avec un bagage technique (développement Python/JavaScript), amenée à modifier le code, ajouter un scénario, ou déboguer un comportement.*

**Mis à jour le 8 octobre 2026 — Lyan Aljendi**

---

## Table des matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Carte du code](#2-carte-du-code)
    - 2 bis. [Les prompts : `modules.txt`](#2-bis-les-prompts--modulestxt)
3. [Stockage des sessions (SQLite)](#3-stockage-des-sessions-sqlite)
4. [Déploiement](#4-déploiement)
5. [Clés d'API](#5-clés-dapi)
6. [Ajouter un nouveau module (scénario)](#6-ajouter-un-nouveau-module-scénario)
7. [Modifier un persona existant sans le casser](#7-modifier-un-persona-existant-sans-le-casser)
8. [Contact](#8-contact)

---

## 1. Vue d'ensemble de l'architecture

ChatCom est une application Flask (`app.py`) qui sert une seule page frontend (`index.html`, JS vanilla, pas de framework) et orchestre trois services externes :

| Service | Rôle | Appelé depuis |
|---|---|---|
| **Mistral AI** | Génère les réponses du patient (persona) et le texte de l'évaluation automatique | `mistral_client.chat.complete(...)` dans `/chat_stream` et `/evaluate` |
| **Inworld** | Synthèse vocale (texte → audio) des réponses du patient | `stream_opus_chunks()`, appelle `INWORLD_TTS_URL` |
| **Vosk** | Reconnaissance vocale (audio → texte) de ce que dit l'étudiant·e | `transcribe_audio()`, modèle chargé en local, aucun appel réseau |

### Flux d'un message texte

```
Étudiant·e tape un message
  → POST /chat_stream { session_id, message }
  → l'historique de la session est récupéré (sessions[session_id])
  → mistral_client.chat.complete(messages=history)
  → la réponse est découpée en phrases (split_sentences)
  → pour chaque phrase : appel à Inworld (stream_opus_chunks) → audio renvoyé en streaming (ndjson)
  → l'historique mis à jour est sauvegardé (sessions.save_history)
```

### Flux d'un message vocal

```
Étudiant·e parle (bouton micro)
  → audio enregistré côté client (MediaRecorder)
  → POST /transcribe (audio brut)
  → ffmpeg convertit l'audio, Vosk le transcrit en texte
  → le texte transcrit est ensuite envoyé à /chat_stream comme un message normal (flux ci-dessus)
```

---

## 2. Carte du code

### `app.py`

| Élément | Ce qu'il contient |
|---|---|
| `load_interview_modules()` | Lit et valide le fichier `modules.txt` (voir section 2 bis) et renvoie le dictionnaire des scénarios. Appelée **une seule fois, au démarrage**. |
| `INTERVIEW_MODULES` | Dictionnaire central, **construit à partir de `modules.txt`** (il n'est plus écrit à la main dans `app.py`) : une entrée par scénario (ex. `"agressif"`, `"incertitude_espoir"`). Chaque entrée contient `label`, `system` (le persona), `eval` (la grille d'évaluation), `practitioner_label`, `patient_label`, `voice_id`. **L'ordre des modules dans `modules.txt` détermine l'ordre d'affichage dans l'interface.** |
| `VOICES` | Mappe un nom de voix court vers l'identifiant réel Inworld (évite de taper un ID brut dans chaque module) |
| Routes Flask | `/`, `/modules`, `/start_session`, `/transcribe`, `/chat_stream`, `/end_session`, `/evaluate`, `/history`, `/reset`, `/debug_last_error` |
| `_SessionStore` | Classe remplaçant un simple dict Python pour stocker les sessions (voir section 3) |
| `transcribe_audio()` | Convertit l'audio via ffmpeg puis transcrit via Vosk |
| `stream_opus_chunks()` | Appelle l'API Inworld et stream l'audio résultant |

```python
VOICES = {
    "female_fr": "Hélène",
    "male_fr":   "Étienne",
    "male_fr_older": "Alain",
}
```

### `modules.txt`

Fichier texte placé **à côté de `app.py`** (même dossier du dépôt) : il contient tous les prompts (persona + évaluation) ainsi que les libellés et la voix de chaque module. Détails en section 2 bis.

### Fichiers statiques (`static/`)

| Fichier | Rôle |
|---|---|
| `research-unit-logo.png` | Logo de l'Unité de Recherche, affiché sur la page d'accueil (PNG à fond transparent) |
| `fsm-logo.png` | Logo FSM, affiché à côté du badge ULB (en bas à droite) sur toutes les pages (PNG à fond transparent) |
| portraits des patient·e·s | Images référencées par `PATIENT_FICHES` (`portrait`) |

Si un logo est absent, l'image se masque d'elle-même (`onerror`) : la page reste utilisable.

### `index.html`

| Élément | Ce qu'il contient |
|---|---|
| Page d'accueil (`#welcome-page`) | Page de bienvenue (logo de l'Unité de Recherche, message d'accueil, Prof. Jennifer Foucart, bouton de démarrage), affichée avant le choix de l'entretien. Le crédit de l'ingénieure (`#welcome-fine`) est volontairement discret, en bas à gauche. Enchaînement : accueil → choix de l'entretien (bouton « ← retour » vers l'accueil) → fiche patient → conversation → transcription. |
| `PATIENT_FICHES` | Objet JS, une entrée par module (même clé que dans `modules.txt`), définissant le contenu HTML de la fiche patient affichée avant de démarrer une consultation |
| `sendMessage()` | Envoie un message texte/transcrit à `/chat_stream`, gère le streaming audio de la réponse |
| `startCall()` / `stopCall()` | Gestion du micro (MediaRecorder, détection de silence) |
| `PATIENT_FICHES` et `modules.txt` sont indépendants mais **doivent partager les mêmes clés** (ex. `"agressif"`) — rien ne le vérifie automatiquement, une faute de frappe dans l'un des deux casse silencieusement l'affichage de la fiche pour ce module. |

```javascript
const PATIENT_FICHES = {
    agressif: {
        name: "Valérie Decocq",
        portrait: "/static/valerie.jpg",
        html: `
            <div class="fiche-header">
                <div class="fiche-tag">Fiche Patiente</div>
                <div class="fiche-title">Valérie Decocq</div>
            </div>
            <div class="fiche-section">
                <div class="fiche-section-title">Identité &amp; Contexte</div>
                <p>40 ans, en couple, mère de deux jeunes enfants (2 et 6 ans).<br />
                Responsable de communication dans une société de transport.</p>
            </div>
            <div class="fiche-section">
                <div class="fiche-section-title">Motif de consultation</div>
                <p>Sur référence de son médecin généraliste, pour des douleurs
                diffuses et migratrices depuis environ 3 mois, sans cause identifiée.</p>
            </div>
            <div class="fiche-section">
                <div class="fiche-section-title">Parcours médical</div>
                <p style="margin-bottom:8px">Déjà consultés, sans solution :</p>
                <ul>
                    <li>Médecin généraliste (radios normales, conseil de bouger
                    davantage, déjà essayé sans succès)</li>
                    <li>Homéopathe (suppression du lactose proposée, sans effet)</li>
                    <li>Ostéopathe (l'a renvoyée consulter ici)</li>
                </ul>
            </div>
        `
    },
    // ... une entrée par module, même clé que dans modules.txt
};
```

---

## 2 bis. Les prompts : `modules.txt`

Les prompts des patient·e·s et des grilles d'évaluation ne sont **plus dans `app.py`** : ils sont dans `modules.txt`, lu une seule fois au démarrage de l'application par `load_interview_modules()`. Le contenu des prompts est strictement identique à l'ancien dictionnaire ; seul leur emplacement a changé, ce qui permet de modifier un persona sans toucher au code Python.

### Format du fichier

Un module = un bloc délimité par des lignes de marqueurs :

```
=== MODULE ma_cle ===
label: Nom affiché dans l'interface
practitioner_label: Soignant
patient_label: Patiente
voice: female_fr
=== SYSTEM ===
RÔLE : Tu es [Nom], [...]
[... prompt du patient ...]
=== EVAL ===
Cadre [NOM DU FRAMEWORK]
[... grille d'évaluation ...]
=== END MODULE ===
```

- `ma_cle` : identifiant du module (même clé que dans `PATIENT_FICHES`).
- `voice` : **clé** du dictionnaire `VOICES` (`female_fr`, `male_fr`, `male_fr_older`), pas l'identifiant Inworld brut ; `app.py` le convertit en `voice_id`.
- Des lignes de commentaire commençant par `#` sont permises **avant le premier module** (le fichier en contient une en en-tête qui rappelle ce format).
- Les fins de ligne Windows (CRLF) sont tolérées.
- **L'ordre des blocs = l'ordre d'affichage** dans l'interface.

### Validation au démarrage (échec explicite)

`load_interview_modules()` refuse un fichier mal formé en levant une `RuntimeError` : marqueur manquant ou imbriqué (par ex. `=== END MODULE ===` oublié), champ d'en-tête absent, clé de module en double, voix inconnue, section vide, fichier introuvable. Dans ce cas l'application **ne démarre pas** ; sur Render, le déploiement échoue et **la version précédente reste en ligne**, ce qui évite de publier un fichier cassé. La cause exacte est écrite dans les logs de déploiement.

Au démarrage réussi, une ligne est écrite dans les logs :

```
[MODULES] 6 modules chargés depuis modules.txt: communication_generale, motivational, agressif, ...
```

Vérifier cette ligne après chaque déploiement qui touche aux prompts.

### Conséquences pratiques

- **Modifier un prompt = modifier `modules.txt` puis redéployer** (le fichier n'est lu qu'au démarrage : une modification sans redéploiement n'a aucun effet sur le service en ligne).
- **`modules.txt` doit être commité dans le dépôt, à côté de `app.py`.** Oublier de le pousser fait échouer le démarrage (« fichier introuvable »).
- **Performance** : aucun impact. Le fichier (~100 Ko) est lu une fois au démarrage ; ensuite les prompts sont en mémoire, exactement comme avant. Aucune lecture de fichier n'a lieu pendant les conversations.

---

## 3. Stockage des sessions (SQLite)

Les sessions ne sont **plus** stockées dans un simple dict Python en mémoire. C'était le cas initialement, mais un dict en mémoire est perdu dès que le processus Python redémarre — notamment lors d'un `WORKER TIMEOUT` gunicorn (le worker est tué et remplacé par un nouveau processus, mémoire vide). Cela provoquait des erreurs `"Session not found"` pour tous les utilisateurs en cours de conversation au moment du redémarrage.

La solution actuelle : un fichier SQLite (`sessions.db`, créé à côté de `app.py`) qui survit à un redémarrage de worker (même conteneur, même système de fichiers), tout en se réinitialisant naturellement lors d'un vrai redéploiement (nouveau conteneur).

```python
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
```

### Point d'attention pour toute modification future

`sessions[session_id]` renvoie une **copie fraîche** à chaque appel (désérialisée depuis le JSON stocké), **pas une référence vivante** comme le ferait un vrai dict. Concrètement : faire `history = sessions[id]["history"]` puis `history.append(...)` ne sauvegarde rien tant que `sessions.save_history(id, history)` n'est pas appelé explicitement. C'est déjà géré correctement dans `/chat_stream` (seul endroit qui mute l'historique), mais toute nouvelle route qui modifierait l'historique devra faire le même appel explicite.

```python
# Extrait de /chat_stream — le seul endroit qui mute history
session_row    = sessions[session_id]
history        = session_row["history"]
interview_type = session_row.get("interview_type")
...
history.append({"role": "user", "content": user_text})

try:
    response = mistral_client.chat.complete(model=MISTRAL_MODEL, messages=history)
    assistant_reply = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_reply})
    sessions.save_history(session_id, history)   # ← sauvegarde explicite requise
except Exception as e:
    history.pop()   # rien n'a été sauvegardé, donc annuler localement suffit
    ...
```

---

## 4. Déploiement

Render exécute l'application via la **Start Command** suivante (visible dans Settings → Start Command sur le service) :

```
gunicorn app:app --timeout 120 --threads 16
```

Points importants :
- **Aucun `--workers` n'est précisé** → gunicorn tourne avec **un seul processus (worker)**, qui charge le modèle Vosk une seule fois en mémoire.
- **`--threads 16`** : ce worker traite jusqu'à **16 requêtes en parallèle** (threads) au lieu d'une seule à la fois. Sans cela, un seul étudiant en cours de transcription (`/transcribe`, ffmpeg + Vosk) bloquait tous les autres. Ce réglage est compatible avec le stockage des sessions (section 3) : chaque opération SQLite ouvre sa propre connexion (rien n'est partagé entre threads), la base est en mode WAL et le délai d'attente est de 10 s.
- **`--timeout 120`** (au lieu des 30s par défaut de gunicorn) : une requête qui dépasse ce délai fait tuer le worker par gunicorn (`WORKER TIMEOUT` dans les logs), ce qui redémarre le processus. Avant la mise en place du stockage SQLite (section 3), cela effaçait toutes les sessions actives.
- **Fichiers à déployer ensemble** : `app.py`, `modules.txt`, `index.html` et le dossier `static/` (logos, portraits) doivent être dans le dépôt ; mettre à jour l'un sans l'autre (ex. `app.py` récent sans `modules.txt`) empêche le démarrage ou casse l'affichage.
- Si Render semble servir d'anciens fichiers après une mise à jour, utiliser Manual Deploy → **Clear build cache & deploy**.
- Le déploiement se fait via Render, connecté au dépôt GitHub du projet — un `git push` sur la branche suivie déclenche normalement un redéploiement automatique (à vérifier dans Settings → Build & Deploy selon la configuration actuelle du service).

**Estimation de charge** : le travail réellement coûteux en CPU est `/transcribe` (ffmpeg + Vosk) ; les appels Mistral/Inworld sont de l'attente réseau, pas du calcul local. Avec `--threads 4`, plusieurs étudiant·e·s peuvent être transcrit·e·s en même temps sans file d'attente (le service dispose de 2 CPU). Pour ~20 étudiant·e·s en usage simultané, cette configuration est suffisante.

Points de vigilance liés aux threads :
- **Deux requêtes simultanées sur la *même* session** : `save_history` réécrit l'historique complet, donc la dernière requête écrase la précédente. En usage normal les requêtes d'une même session sont séquentielles (l'étudiant parle, puis attend la réponse), mais un double clic pourrait déclencher ce cas.
- **Connexion Inworld partagée** : `inworld_session` (`requests.Session`) est utilisée par tous les threads. Cela fonctionne en général ; en cas d'erreurs `[TTS ERROR]` après le passage aux threads, c'est le premier point à vérifier.

---

## 5. Clés d'API

Les deux clés nécessaires :

```python
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "...")
INWORLD_API_KEY = os.environ.get("INWORLD_API_KEY", "...")
```

Elles sont lues depuis les variables d'environnement (`$MISTRAL_API_KEY`, `$INWORLD_API_KEY`), configurées sur Render dans l'onglet **Environment** du service.

---

## 6. Ajouter un nouveau module (scénario)

Un module = un bloc dans `modules.txt` **et** une entrée correspondante dans `PATIENT_FICHES` (`index.html`), avec **la même clé** dans les deux fichiers. Aucune modification de `app.py` n'est nécessaire.

### 6.1 — Dans `modules.txt`

Ajouter un bloc à la fin du fichier (ou à l'endroit voulu : l'ordre des blocs = l'ordre d'affichage) :

```
=== MODULE ma_nouvelle_cle ===
label: Nom affiché dans l'interface
practitioner_label: Soignant
patient_label: Patient
voice: female_fr
=== SYSTEM ===
RÔLE : Tu es [Nom], un·e patient·e. [...]

IDENTITÉ :
[...]

SITUATION :
[...]

ÉTAT DE DÉPART : [NOM_ETAT_INITIAL]

---

[NOM_ETAT_INITIAL]
Quand : [...]
Réponses : [...]
Ton : [...]
Exemples : [...]
Transitions :
  - [...] → [AUTRE_ETAT]

---

[... autres états ...]

---

RÉACTIONS AUX APPROCHES DU SOIGNANT :
[...]

---

RÈGLES ABSOLUES :
1. Tu joues UNIQUEMENT le/la patient·e. [...]
2. TON PREMIER MESSAGE : commence par « Bonjour », puis réponds
directement à ce que le soignant vient de te dire ou de te demander [...]
3. [...]
=== EVAL ===
Cadre [NOM DU FRAMEWORK]
[...]

N'ajoutez aucun format à votre réponse, uniquement du texte brut.
=== END MODULE ===
```

Points d'attention :
- `voice` prend une **clé de `VOICES`** (`female_fr`, `male_fr` ou `male_fr_older`) ; pour ajouter une nouvelle voix, l'ajouter d'abord à `VOICES` dans `app.py`.
- Les quatre champs d'en-tête (`label`, `practitioner_label`, `patient_label`, `voice`) sont obligatoires.
- Les marqueurs `=== ... ===` doivent être seuls sur leur ligne et **ne jamais apparaître à l'intérieur d'un prompt**.
- Ne rien écrire entre un `=== END MODULE ===` et le `=== MODULE ... ===` suivant.
- Après redéploiement, vérifier dans les logs la ligne `[MODULES] N modules chargés ...` : N doit avoir augmenté de 1.

### 6.2 — Dans `index.html`

Ajouter une entrée dans `PATIENT_FICHES` avec la **même clé exacte** :
```javascript
ma_nouvelle_cle: {
    name: "Nom du patient",
    portrait: "/static/nom_image.jpg",
    html: `
        <div class="fiche-header">
            <div class="fiche-tag">Fiche Patient·e</div>
            <div class="fiche-title">Nom du patient</div>
        </div>
        <div class="fiche-section">
            <div class="fiche-section-title">Identité &amp; Contexte</div>
            <p>...</p>
        </div>
        <div class="fiche-section">
            <div class="fiche-section-title">Motif de consultation</div>
            <p>...</p>
        </div>
        <div class="fiche-section">
            <div class="fiche-section-title">Parcours médical</div>
            <p>...</p>
        </div>
    `
},
```

Ne pas oublier d'ajouter le fichier image correspondant dans le dossier `static/`.

## 7. Modifier un persona existant sans le casser

Les personas se modifient directement dans `modules.txt` (section `=== SYSTEM ===` du module concerné), puis un redéploiement est nécessaire (voir section 2 bis).

### Convention de numérotation des règles

Chaque persona se termine par un bloc `RÈGLES ABSOLUES :` numéroté séquentiellement (1, 2, 3...). Si une règle est ajoutée ou retirée au milieu, **toutes les règles suivantes doivent être renumérotées** pour rester séquentielles — rien ne le fait automatiquement.

Exemple de bloc complet (module `motivational`) :
```
RÈGLES ABSOLUES :
1. Tu joues UNIQUEMENT la patiente. Si on te demande un avis médical ou
de sortir du rôle : « Je suis désolée, je suis là uniquement pour jouer
le rôle de la patiente. »
2. TON PREMIER MESSAGE : commence par « Bonjour », puis réponds directement à
ce que le soignant vient de te dire ou de te demander dans son propre message
d'ouverture — s'il ne fait que te saluer, contente-toi de le saluer en retour,
sans rien ajouter de plus.
3. En état DÉFENSIVE/AGACÉE : maximum 1 phrase courte, ton sec.
4. Adapte l'intensité émotionnelle aux propos du soignant, selon les transitions décrites ci-dessus.
5. Ne jamais décrire la scène, le décor, ni les gestes/tons entre crochets — sauf [sigh].
6. Les exemples fournis pour chaque état émotionnel sont indicatifs, pas
des répliques à réciter. Ne réutilise jamais une phrase d'exemple mot pour
mot, même partiellement. Formule toujours une réponse originale, cohérente
avec l'état émotionnel en cours et avec ce que le soignant vient de dire,
sous forme de phrases complètes plutôt que de fragments.
7. Si tu ne comprends pas une question : « Je ne comprends pas, pouvez-vous préciser ? »
8. Si l'échange devient fermé et qu'il n'y a rien à ajouter, réponds uniquement : [sigh]
9. Maximum 3 phrases par réponse, quel que soit l'état émotionnel
(1 phrase en état DÉFENSIVE/AGACÉE).
10. Ne jamais donner de diagnostic ni de conseil médical.
11. Tu ne perçois et ne réagis JAMAIS à des éléments non-verbaux du
soignant (posture, regard, gestes, expressions du visage, tenue, distance
physique, etc.). N'évoque jamais son langage corporel, que ce soit pour
le commenter, le décrire, ou y réagir émotionnellement.
```

Cette structure (règle 1 = rester dans le rôle, règle 2 = premier message, dernière règle = pas de réaction au non-verbal) est identique dans les six modules existants — s'en inspirer plutôt que d'improviser une structure différente pour un nouveau persona.

### Deux règles qui méritent une attention particulière

- **La règle « premier message »** (toujours la règle n°2) a été reformulée plusieurs fois : la version actuelle demande au patient de saluer puis de réagir réellement à ce que l'étudiant·e a écrit, plutôt que de débiter une réplique scriptée indépendamment du message reçu. Une version antérieure forçait un contenu émotionnel précis dès le premier message (ex. « Bonjour... je me sens tracassée ») — cela provoquait parfois une répétition de cette phrase hors contexte plus tard dans la conversation, probablement parce que le même contenu émotionnel était dupliqué à la fois dans cette règle et dans la description de l'état de départ. Éviter de dupliquer le même exemple à deux endroits du prompt.
- **La règle « ne pas réciter les exemples mot pour mot »** (règle 6 dans l'exemple ci-dessus) inclut maintenant explicitement *« sous forme de phrases complètes plutôt que de fragments »* — ajouté après qu'un module produisait des réponses trop télégraphiques en reprenant le style très court de ses exemples au lieu de formuler une phrase complète adaptée.

### Tester un changement de prompt

Il n'existe pas de suite de tests automatisés sur le contenu des personas (c'est un texte en langage naturel interprété par le LLM, pas du code à proprement parler). La validation se fait par conversation manuelle : dérouler un échange complet après toute modification d'un persona, en particulier autour des transitions d'état modifiées.

---

## 8. Contact

**Lyan Aljendi**
lyan.aljendi@gmail.com
