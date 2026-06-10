from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import requests
import re
import os
import time
import asyncio
from dotenv import load_dotenv
from retrieve import configurer_chatbot, poser_question_avec_memoire
from filters import is_rate_limited, handle_trivial, SPAM_REPLY

load_dotenv()

app = FastAPI()

# ---------------------------------------------------------------------------
# Variables d'environnement WhatsApp
# ---------------------------------------------------------------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WA_API_VERSION = os.getenv("WA_API_VERSION", "v19.0")

if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID or not VERIFY_TOKEN:
    raise ValueError("Variables d'environnement WhatsApp manquantes dans le fichier .env")

# ---------------------------------------------------------------------------
# Initialisation du Chatbot
# ---------------------------------------------------------------------------
try:
    chatbot = configurer_chatbot()
    print("[INIT] ✅ Chatbot HAAC prêt !")
except Exception as e:
    print(f"[INIT] ❌ Erreur lors du chargement du chatbot : {e}")
    raise

# ---------------------------------------------------------------------------
# Structures de données globales (Mémoire Vive Locale)
# ---------------------------------------------------------------------------
processed_message_ids: set[str] = set()
MAX_PROCESSED_IDS = 2000

last_message_time: dict[str, float] = {}
followup_tasks: dict[str, asyncio.Task] = {}

# ---------------------------------------------------------------------------
# Configurations du Call Center HAAC
# ---------------------------------------------------------------------------
FOLLOWUP_DELAY = 600  # 10 minutes en secondes

FOLLOWUP_MESSAGE = (
    "Hello! Je remarque que vous n'avez pas envoyé de message depuis un moment. "
    "Avez-vous d'autres préoccupations concernant la réglementation des médias sur lesquelles je peux vous aider ? 😊"
)

# On ne garde que les mots isolés ultra-simples pour économiser l'API
GOODBYE_TRIGGERS = ["non", "merci", "au revoir", "bye", "stop", "no", "nothing"]



GOODBYE_MESSAGE = (
    "C'est un plaisir de vous avoir assisté ! La HAAC vous remercie pour votre confiance. 🙏\n\n"
    "Nous restons à votre entière disposition pour toute autre préoccupation.\n"
    "Pour plus d'informations, visitez notre portail ou écrivez-nous :\n"
    "📧 contact@haac.bj\n"
    "🌐 https://haac.bj\n\n"
    "L'équipe d'assistance HAAC vous souhaite une excellente journée ! 😊"
)


def check_if_goodbye_llm(llm, text: str) -> bool:
    """Utilise le LLM pour détecter si l'utilisateur souhaite clore la conversation."""
    prompt = f"""Analyse le message court d'un utilisateur de chatbot et détermine s'il exprime la fin de la discussion (intention de dire au revoir, de remercier pour clore, ou d'indiquer qu'il n'a plus de questions).

MESSAGE DE L'UTILISATEUR : "{text}"

Réponds UNIQUEMENT par le mot OUI si l'utilisateur veut clore la discussion.
Réponds UNIQUEMENT par le mot NON si l'utilisateur pose une question ou attend une suite.

Exemples de OUI : "non c'est bon merci", "c'est tout pour moi", "merci bien", "fin", "plus de questions", "merci bonsoir".
Exemples de NON : "non, je veux plutôt savoir...", "merci mais qu'en est-il de...", "c'est bon pour la carte, et pour la radio ?".

Réponse (OUI ou NON) :"""
    try:
        response = llm.invoke(prompt).content.strip().upper()
        return "OUI" in response
    except Exception as e:
        print(f"[CLÔTURE] ⚠️ Erreur LLM classification clôture : {e}")
        return False


class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None

# ---------------------------------------------------------------------------
# Outils de formatage et d'envoi
# ---------------------------------------------------------------------------
def markdown_to_whatsapp(text: str) -> str:
    text = text.replace("**", "*")
    text = re.sub(r'#{1,6}\s*(.+)', r'*\1*', text)
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def send_whatsapp_message(to: str, text: str):
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
            print(f"[WHATSAPP] ✅ Message envoyé avec succès à {to}")
        else:
            print(f"[WHATSAPP] ❌ Échec envoi — Status: {response.status_code} — {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[WHATSAPP] ❌ Erreur réseau : {e}")

# ---------------------------------------------------------------------------
# Système de relance asynchrone (10 minutes)
# ---------------------------------------------------------------------------
async def wait_and_send_followup(sender_id: str):
    """Attend la durée réglementaire puis envoie une relance si aucune activité n'a eu lieu."""
    try:
        await asyncio.sleep(FOLLOWUP_DELAY)
        # On vérifie si l'utilisateur a écrit entre-temps
        current_time = time.time()
        elapsed = current_time - last_message_time.get(sender_id, 0)
        
        if elapsed >= FOLLOWUP_DELAY:
            send_whatsapp_message(sender_id, FOLLOWUP_MESSAGE)
            print(f"[FOLLOWUP] 📤 Relance call center transmise à {sender_id}")
            # On met à jour pour éviter une relance immédiate en boucle
            last_message_time[sender_id] = current_time
    except asyncio.CancelledError:
        # La tâche a été annulée parce que l'utilisateur a répondu avant les 10 minutes
        pass

# ---------------------------------------------------------------------------
# Traitement lourd en tâche de fond (RAG + API)
# ---------------------------------------------------------------------------
async def process_whatsapp_pipeline(sender_id: str, user_text: str):
    """Exécute l'analyse et la recherche documentaire sans bloquer la route principale."""
    try:
        # 1. Filtre Anti-Spam / Rate limit
        if is_rate_limited(sender_id):
            send_whatsapp_message(sender_id, SPAM_REPLY)
            return

        cleaned_text = user_text.strip().lower().replace(".", "").replace("!", "")
        
       # 2. Clôture de discussion explicite ou sémantique
        is_goodbye = (cleaned_text in GOODBYE_TRIGGERS) or check_if_goodbye_llm(chatbot["llm"], user_text)
        
        if is_goodbye:
            send_whatsapp_message(sender_id, GOODBYE_MESSAGE)
            print(f"[CALL-CENTER] 🚪 Fin de session validée pour {sender_id}")
            # Annuler la relance en cours car la discussion est close
            if sender_id in followup_tasks:
                followup_tasks[sender_id].cancel()
            return

        # --- CORRECTIF : EXTRACTION DE L'HISTORIQUE POUR LE FILTRE ---
        # On va chercher l'historique de l'utilisateur dans la mémoire du chatbot
        history = chatbot["memory"].get_formatted_history(sender_id)

        # 3. Filtre messages triviaux (On passe maintenant l'argument 'history')
        trivial_response = handle_trivial(user_text, llm=chatbot["llm"], history=history)
        if trivial_response:
            send_whatsapp_message(sender_id, trivial_response)
            print(f"[FILTER] 💬 Réponse triviale envoyée à {sender_id}")
            # Planifier la relance même après un message trivial
            reset_followup_timer(sender_id)
            return

        # 4. Pipeline RAG complet (FAISS + Tavily)
        print(f"[PIPELINE] 🔍 Analyse de la requête pour {sender_id}...")
        t0 = time.time()
        result = poser_question_avec_memoire(chatbot, user_text, user_id=sender_id)
        print(f"[PIPELINE] ✅ Synthèse achevée en {time.time() - t0:.2f}s")

        bot_answer = markdown_to_whatsapp(result['response'])
        send_whatsapp_message(sender_id, bot_answer)
        
        # 5. Gestion des timers du Call Center
        reset_followup_timer(sender_id)

    except Exception as e:
        print(f"[PIPELINE-ERREUR] ❌ Dysfonctionnement : {e}")
        send_whatsapp_message(sender_id, "Navré, mes systèmes rencontrent une surcharge temporaire. Veuillez reformuler.")


def reset_followup_timer(sender_id: str):
    """Annule l'ancienne planification de relance et en crée une nouvelle."""
    last_message_time[sender_id] = time.time()
    if sender_id in followup_tasks and not followup_tasks[sender_id].done():
        followup_tasks[sender_id].cancel()
    
    followup_tasks[sender_id] = asyncio.create_task(wait_and_send_followup(sender_id))

# ---------------------------------------------------------------------------
# Endpoints Webhooks (Entrées API)
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        print("[WEBHOOK] ✅ Handshake Meta validé")
        return Response(content=params.get("hub.challenge"), status_code=200)
    return Response(content="Verification failed", status_code=403)

@app.post("/webhook")
async def handle_message(request: Request):
    data = await request.json()
    
    # Réponse immédiate préparée pour libérer WhatsApp en moins de 500ms
    immediate_response = JSONResponse(content={"status": "accepted"}, status_code=200)

    try:
        entry = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
        if not entry or 'messages' not in entry:
            return immediate_response

        message = entry['messages'][0]
        if message.get('type') != 'text':
            return immediate_response

        message_id = message.get('id', '')

        # Déduplication atomique
        if message_id in processed_message_ids:
            print(f"[ANTI-RETRY] 🛑 Répétition évitée pour l'ID: {message_id}")
            return immediate_response

        processed_message_ids.add(message_id)
        if len(processed_message_ids) > MAX_PROCESSED_IDS:
            processed_message_ids.pop() # Élimine le plus ancien

        sender_id = message['from']
        user_text = message['text']['body']

        print(f"\n[WEBHOOK] 📩 Nouveau message de : {sender_id} (ID: {message_id})")

        # Propulsion de la tâche en arrière-plan
        asyncio.create_task(process_whatsapp_pipeline(sender_id, user_text))

    except Exception as e:
        print(f"[WEBHOOK-ERREUR] ❌ Erreur critique parsing : {e}")

    return immediate_response

@app.post("/api/ask")
async def ask_question(request_data: QuestionRequest):
    # Conserver ton endpoint Web de test intact
    user_id = request_data.user_id or f"web_user_{int(time.time())}"
    result = poser_question_avec_memoire(chatbot, request_data.question, user_id=user_id)
    return {"response": result['response'], "sources": result['sources']}