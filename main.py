from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import requests
import re
import os
import time
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

if not WHATSAPP_TOKEN:
    raise ValueError("WHATSAPP_TOKEN manquant dans le fichier .env")
if not PHONE_NUMBER_ID:
    raise ValueError("PHONE_NUMBER_ID manquant dans le fichier .env")
if not VERIFY_TOKEN:
    raise ValueError("VERIFY_TOKEN manquant dans le fichier .env")

# ---------------------------------------------------------------------------
# Chargement du chatbot
# ---------------------------------------------------------------------------

try:
    chatbot = configurer_chatbot()
    print("[INIT] ✅ Chatbot prêt !")
except Exception as e:
    print(f"[INIT] ❌ Erreur lors du chargement du chatbot : {e}")
    raise

# ---------------------------------------------------------------------------
# Modèles Pydantic
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None

# ---------------------------------------------------------------------------
# Formatage WhatsApp
# ---------------------------------------------------------------------------

def markdown_to_whatsapp(text: str) -> str:
    text = text.replace("**", "*")
    text = re.sub(r'#{1,6}\s*(.+)', r'*\1*', text)
    text = re.sub(r'^\s*[\*\-]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Envoi WhatsApp
# ---------------------------------------------------------------------------

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
            print(f"[WHATSAPP] ✅ Message envoyé à {to}")
        else:
            print(f"[WHATSAPP] ❌ Échec — Status: {response.status_code} — {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[WHATSAPP] ❌ Erreur réseau : {e}")

# ---------------------------------------------------------------------------
# Endpoint API web
# ---------------------------------------------------------------------------

@app.post("/api/ask")
async def ask_question(request_data: QuestionRequest):
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

        trivial_response = handle_trivial(question, llm=chatbot["llm"])
        if trivial_response:
            print(f"[API] 💬 Réponse triviale : '{trivial_response[:60]}'")
            return {
                "response": trivial_response,
                "sources": [],
                "user_id": user_id,
                "status": "success"
            }

        t0 = time.time()
        result = poser_question_avec_memoire(chatbot, question, user_id=user_id)

        print(f"[API] ✅ Réponse générée en {time.time() - t0:.2f}s")
        print(f"[API] 📚 Sources : {', '.join(result['sources'])}")
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
# Webhook WhatsApp
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

        # Filtre 1 : anti-spam
        if is_rate_limited(sender_id):
            send_whatsapp_message(sender_id, SPAM_REPLY)
            print(f"[FILTER] 🚫 Réponse anti-spam envoyée à {sender_id}")
            return {"status": "ok"}

        # Filtre 2 : messages triviaux
        trivial_response = handle_trivial(user_text, llm=chatbot["llm"])
        if trivial_response:
            send_whatsapp_message(sender_id, trivial_response)
            print(f"[FILTER] 💬 Réponse contextuelle envoyée à {sender_id} : '{trivial_response[:60]}'")
            return {"status": "ok"}

        # Pipeline RAG normal
        print(f"[FAISS] 🔍 Recherche des documents pertinents...")
        t0 = time.time()

        result = poser_question_avec_memoire(chatbot, user_text, user_id=sender_id)

        print(f"[FAISS] ✅ Réponse générée en {time.time() - t0:.2f}s")
        print(f"[FAISS] 📚 Sources : {', '.join(result['sources'])}")
        print(f"[LLM] 📝 Aperçu : {result['response'][:200]}...")

        bot_answer = markdown_to_whatsapp(result['response'])
        send_whatsapp_message(sender_id, bot_answer)

        print(f"[DONE] ✅ Traitement terminé en {time.time() - t0:.2f}s")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"[ERREUR] ❌ {type(e).__name__} — {e}")
        if sender_id:
            send_whatsapp_message(sender_id, "Désolé, une erreur s'est produite. Veuillez réessayer.")

    return {"status": "ok"}