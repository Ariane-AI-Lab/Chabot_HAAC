from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import requests
import re
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from retrieve import configurer_chatbot, poser_question_avec_memoire

load_dotenv()

app = FastAPI()

#----------------------------------------------------------------------------
# Chargement des variables d'environnement pour WhatsApp
#----------------------------------------------------------------------------

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WA_API_VERSION = os.getenv("WA_API_VERSION", "v19.0")

if not WHATSAPP_TOKEN:
    raise ValueError("WHATSAPP_TOKEN manquant dans le fichier .env")
if not PHONE_NUMBER_ID:
    raise ValueError("PHONE_NUMBER_ID manquant dans le fichier .env")
if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN manquant dans le fichier .env")


#----------------------------------------------------------------------------
# CONSTANTES
#----------------------------------------------------------------------------

    # Limites pour la détection de spam (nombre de messages et fenêtre temporelle)
SPAM_MAX_MESSAGES = 5       
SPAM_WINDOW_SECONDS = 60    

SPAM_REPLY = (
    "⚠️ Vous envoyez des messages trop rapidement. "
    "Merci de patienter quelques instants avant de poser votre prochaine question."
)


    # Mots-clés et patterns qui signalent un message trivial / hors-sujet
TRIVIAL_PATTERNS = [
    r"^\s*(bonjour|bonsoir|salut|hello|hi|hey|coucou|allo|allô)\s*[!?.]*\s*$",
    r"^\s*(merci|thanks|thx|thank you|ok merci|super merci)\s*[!?.]*\s*$",
    r"^\s*(ok|okay|d'accord|dacord|oui|non|yes|no|👍|👎|😊|🙏)\s*$",
    r"^\s*.{0,2}\s*$",
    r"^\s*(test|testing|essai|123|ping)\s*[!?.]*\s*$",
]

TRIVIAL_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRIVIAL_PATTERNS]

TRIVIAL_REPLY = (
    "👋 Salut ! Je suis l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel "
    "et de la Communication du Bénin).\n\n"
    "Je suis ici pour répondre à vos questions sur la réglementation audiovisuelle, "
    "les textes officiels et les procédures de la HAAC.\n\n"
    "N'hésitez pas à me poser votre question ! 😊"
)


#----------------------------------------------------------------------------
# CONFIGURATION DU CHATBOT
#----------------------------------------------------------------------------

try:
    chatbot = configurer_chatbot()
    print("[INIT] ✅ Chatbot prêt !")
except Exception as e:
    print(f"[INIT] ❌ Erreur lors du chargement du chatbot : {e}")
    raise


# ---------------------------------------------------------------------------
# MODÈLES PYDANTIC POUR LES REQUÊTES API
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Modèle pour les requêtes de questions à l'API"""
    question: str
    user_id: str | None = None  # Optionnel : ID utilisateur pour la mémoire de conversation


# ---------------------------------------------------------------------------
# ANTI-SPAM : Nombre limites de messages par utilisateurs et fenêtre de temps
# ---------------------------------------------------------------------------

# Stocke les timestamps des derniers messages par utilisateur
user_message_times: dict[str, list[float]] = defaultdict(list)

def is_rate_limited(sender_id: str) -> bool:
    now = time.time()
    times = [t for t in user_message_times[sender_id] if now - t < SPAM_WINDOW_SECONDS]
    user_message_times[sender_id] = times

    if len(times) >= SPAM_MAX_MESSAGES:
        print(f"[SPAM] 🚫 {sender_id} bloqué ({len(times)} msgs en {SPAM_WINDOW_SECONDS}s)")
        return True

    user_message_times[sender_id].append(now)
    return False

# ---------------------------------------------------------------------------
# FILTRAGE DES MESSAGES TRIVIAUX
# ---------------------------------------------------------------------------

def is_trivial(text: str) -> bool:
    """Retourne True si le message est une salutation, remerciement ou contenu vide."""
    for pattern in TRIVIAL_COMPILED:
        if pattern.match(text):
            print(f"[FILTER] ⚠️ Message trivial détecté : '{text[:50]}'")
            return True
    return False


# ---------------------------------------------------------------------------
# FORMATAGE WHATSAPP
# ---------------------------------------------------------------------------


# def markdown_to_whatsapp(text: str) -> str:
#     text = text.replace("**", "*")                              # 1. Gras d'abord
#     text = re.sub(r'#{1,6}\s*(.+)', r'*\1*', text)             # 2. Titres
#     text = re.sub(r'_(.+?)_', r'\1', text)                     # 3. Italique
#     text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)  # 4. Listes
#     text = re.sub(r'\n{3,}', '\n\n', text)                     # 5. Espacement
#     return text.strip()


def markdown_to_whatsapp(text: str) -> str:
    text = text.replace("**", "*")                                  # Gras : **x** → *x*
    text = re.sub(r'#{1,6}\s*(.+)', r'*\1*', text)                  # Titres : ## x → *x*
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE) # Listes
    text = re.sub(r'\n{3,}', '\n\n', text)                          # Espacement
    return text.strip()


# ---------------------------------------------------------------------------
# ENVOI WHATSAPP
# ---------------------------------------------------------------------------

def send_whatsapp_message(to: str, text: str):
    """
    Envoie un message texte à un utilisateur via l'API WhatsApp Business (Meta).

    Args:
        to (str): Numéro de téléphone du destinataire au format international
                  sans le '+' (ex: '22960112233').
        text (str): Contenu du message à envoyer.

    Returns:
        None

    Raises:
        requests.exceptions.RequestException: En cas d'erreur réseau (timeout,
                                              connexion refusée, etc.).

    Example:
        >>> send_whatsapp_message("22960112233", "Bonjour, comment puis-je vous aider ?")
    """

    url = f"https://graph.facebook.com/{WA_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"[WHATSAPP] ✅ Message envoyé à {to}")
        else:
            print(f"[WHATSAPP] ❌ Échec — Status: {response.status_code} — {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[WHATSAPP] ❌ Erreur réseau : {e}")


# ---------------------------------------------------------------------------
# ENDPOINT PLATEFORME EN LIGNE 
# ---------------------------------------------------------------------------

@app.post("/api/ask")
async def ask_question(request_data: QuestionRequest):
    """
    Endpoint pour poser des questions au chatbot depuis une plateforme en ligne.

    Args:
        request_data (QuestionRequest): Corps de la requête contenant :
            - question (str): La question à poser au chatbot.
            - user_id (str, optional): ID utilisateur pour maintenir la mémoire
              de conversation. Généré automatiquement si non fourni.

    Returns:
        JSONResponse: Un objet JSON contenant :
            - response (str): La réponse du chatbot.
            - sources (list): Les fichiers sources utilisés pour la réponse.
            - user_id (str): L'ID utilisateur (fourni ou généré).
            - status (str): "success" en cas de succès.

    Raises:
        400: Si la question est vide.
        429: Si l'utilisateur dépasse le seuil de messages autorisés (anti-spam).
        500: En cas d'erreur interne lors du traitement.

    Example:
        >>> POST /api/ask
        >>> {"question": "Quelles sont les obligations d'une radio ?", "user_id": "user_123"}
    """

    user_id = None
    try:
        question = request_data.question.strip()
        user_id = request_data.user_id or f"web_user_{int(time.time() * 1000)}"
        
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "La question ne peut pas être vide", "status": "error"}
            )
        
        if is_rate_limited(user_id):

            return JSONResponse(
                status_code=429,
                content={"error": SPAM_REPLY, "status": "error"}
            )
        
        print(f"\n{'='*50}")
        print(f"[API] 📩 Question reçue de : {user_id}")
        print(f"[API] 💬 Question : {question}")
        
        # Traiter la question avec le chatbot
        print(f"[FAISS] 🔍 Recherche des documents pertinents...")
        t0 = time.time()
        
        result = poser_question_avec_memoire(chatbot, question, user_id=user_id)
        
        t1 = time.time()
        print(f"[FAISS] ✅ Réponse générée en {t1 - t0:.2f}s")
        print(f"[FAISS] 📚 Sources : {', '.join(result['sources'])}")
        print(f"[API] ✅ Traitement terminé en {time.time() - t0:.2f}s")
        print(f"{'='*50}\n")
        
        return {
            "response": result['response'],
            "sources": result['sources'],
            "user_id": user_id,
            "status": "success"
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error"}
        )


# ---------------------------------------------------------------------------
# Vérification Meta
# ---------------------------------------------------------------------------

@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        print("[WEBHOOK] ✅ Vérification Meta réussie")
        return Response(content=params.get("hub.challenge"), status_code=200)
    print("[WEBHOOK] ❌ Échec de vérification Meta")
    return Response(content="Verification failed", status_code=403)


# ---------------------------------------------------------------------------
# Endpoint pour recevoir les messages WhatsApp (webhook)
# ---------------------------------------------------------------------------

@app.post("/webhook")
async def handle_message(request: Request):
    data = await request.json()
    sender_id = None
    try:
        entry = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
        if not entry:
            return {"status": "ok"}
        
        if 'messages' not in entry:
            print("[WEBHOOK] ⚠️ Événement sans message (statut, réaction...) — ignoré")
            return {"status": "ok"}

        message = entry['messages'][0]

        if message.get('type') != 'text':
            print(f"[WEBHOOK] ⚠️ Type '{message.get('type')}' reçu — ignoré")
            return {"status": "ok"}

        sender_id = message['from']
        user_text = message['text']['body']

        print(f"\n{'='*50}")
        print(f"[MESSAGE] 📩 De : {sender_id}")
        print(f"[MESSAGE] 💬 Texte : {user_text}")

        # FILTRE 1 : Anti-spam (rate limiting) 
        if is_rate_limited(sender_id):
            send_whatsapp_message(sender_id, SPAM_REPLY)
            print(f"[FILTER] 🚫 Réponse anti-spam envoyée à {sender_id}")
            return {"status": "ok"}

        # FILTRE 2 : Messages triviaux 
        if is_trivial(user_text):
            send_whatsapp_message(sender_id, TRIVIAL_REPLY)
            print(f"[FILTER] 💬 Réponse triviale envoyée à {sender_id}")
            return {"status": "ok"}

        # PIPELINE NORMAL : RAG + LLM
        print(f"[FAISS] 🔍 Recherche des documents pertinents...")
        t0 = time.time()

        result = poser_question_avec_memoire(chatbot, user_text, user_id=sender_id)

        t1 = time.time()
        print(f"[FAISS] ✅ Réponse générée en {t1 - t0:.2f}s")
        print(f"[FAISS] 📚 Sources : {', '.join(result['sources'])}")
        print(f"[LLM] 📝 Aperçu : {result['response'][:200]}...")

        bot_answer = markdown_to_whatsapp(result['response'])
        send_whatsapp_message(sender_id, bot_answer)

        print(f"[DONE] ✅ Traitement terminé en {time.time() - t0:.2f}s")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"[ERREUR] ❌ {type(e).__name__} — {e}")
        if sender_id:  # déjà disponible
            send_whatsapp_message(sender_id, "Désolé, une erreur s'est produite. Veuillez réessayer.")

    return {"status": "ok"}