import requests
import re
import os
import json
import time
import asyncio
import mammoth
import shutil
import httpx
#import smtplib
#from email.mime.multipart import MIMEMultipart
#from email.mime.text import MIMEText
from datetime import datetime
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import select, update, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import FastAPI, Request, Response, UploadFile, File, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from index import HuggingFaceAPIEmbeddings, preparer_documents
from retrieve import configurer_chatbot, poser_question_avec_memoire
from database import engine, get_db, Base, AsyncSessionLocal
from models import Agent, Conversation, Message, Problematique, Session, MessageIA, MessageSessionIA, SessionIA
from sqlalchemy import select, update, func, and_
from datetime import datetime, timedelta
from filters import (
    is_rate_limited,
    handle_trivial,
    check_if_goodbye_llm,
    SPAM_REPLY,
    FOLLOWUP_DELAY,
    FOLLOWUP_MESSAGE,
    GOODBYE_TRIGGERS,
    GOODBYE_MESSAGE
)
from auth import (
    hasher_mot_de_passe,
    verifier_mot_de_passe,
    creer_token,
    get_agent_connecte,
    get_admin_connecte
)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WA_API_VERSION = os.getenv("WA_API_VERSION", "v19.0")
WABA_ID = os.getenv("WABA_ID")
PATH_FAISS = "faiss_index_haac"
HF_TOKEN = os.getenv("HF_TOKEN")
#GMAIL_SENDER = os.getenv("GMAIL_SENDER")
#GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.getenv("BREVO_SENDER_EMAIL")
SECRET_KEY = os.getenv("SECRET_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_NOM = os.getenv("ADMIN_NOM", "Admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://ton-dashboard.com")


if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID or not VERIFY_TOKEN:
    raise ValueError("Variables d'environnement WhatsApp manquantes dans le fichier .env")


@app.on_event("startup")
async def startup():
    """Crée les tables et le compte admin au premier démarrage."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Agent).where(Agent.email == ADMIN_EMAIL))
        admin_existant = result.scalar_one_or_none()

        if not admin_existant:
            admin = Agent(
                nom=ADMIN_NOM,
                email=ADMIN_EMAIL,
                mot_de_passe=hasher_mot_de_passe(ADMIN_PASSWORD),
                role="admin",
                actif=True
            )
            db.add(admin)
            await db.commit()
            print(f"[INIT] ✅ Compte admin créé : {ADMIN_EMAIL}")
        else:
            print(f"[INIT] ℹ️ Compte admin déjà existant : {ADMIN_EMAIL}")


try:
    chatbot = configurer_chatbot()
    print("[INIT] ✅ Chatbot HAAC prêt !")
except Exception as e:
    print(f"[INIT] ❌ Erreur lors du chargement du chatbot : {e}")
    raise

processed_message_ids: set[str] = set()
MAX_PROCESSED_IDS = 2000
last_message_time: dict[str, float] = {}
followup_tasks: dict[str, asyncio.Task] = {}
SESSIONS_IA_ACTIVES: dict[str, int] = {}
TIMEOUT_TASKS: dict[str, asyncio.Task] = {}
TIMEOUT_CLOTURE = 600  


class QuestionRequest(BaseModel):
    question: str
    user_id: str | None = None


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


async def envoyer_email_html(destinataires: list[str], sujet: str, html_body: str):
    if not BREVO_API_KEY:
        print("[EMAIL] ⚠️ BREVO_API_KEY manquante, envoi ignoré.")
        return False

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={
                    "api-key": BREVO_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "sender": {"name": "HAAC", "email": BREVO_SENDER_EMAIL},
                    "to": [{"email": e} for e in destinataires],
                    "subject": sujet,
                    "htmlContent": html_body,
                },
            )
            if response.status_code in (200, 201):
                print(f"[EMAIL] ✅ Email envoyé à {', '.join(destinataires)}")
            else:
                print(f"[EMAIL] ❌ Échec envoi : {response.status_code} — {response.text}")
        return True
    except Exception as e:
        print(f"[EMAIL] ❌ Échec envoi email : {e}")
        return False
    

async def notifier_agents_par_email(sender_id: str, user_text: str):

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Agent).where(Agent.actif == True))
        agents_actifs = result.scalars().all()
        emails_destinataires = [a.email for a in agents_actifs]

    if not emails_destinataires:
        print("[EMAIL] ⚠️ Aucun agent actif trouvé en base.")
        return

    heure = datetime.now().strftime("%d/%m/%Y à %H:%M")
    dashboard_url = f"{DASHBOARD_URL}/conversation/{sender_id}"

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #d9534f;">🔔 Assistance requise — HAAC Chatbot</h2>
        <table style="border-collapse: collapse; width: 100%;">
            <tr>
                <td style="padding: 8px; font-weight: bold;">Numéro WhatsApp</td>
                <td style="padding: 8px;">+{sender_id}</td>
            </tr>
            <tr style="background: #f9f9f9;">
                <td style="padding: 8px; font-weight: bold;">Message du client</td>
                <td style="padding: 8px;"><em>"{user_text}"</em></td>
            </tr>
            <tr>
                <td style="padding: 8px; font-weight: bold;">Heure</td>
                <td style="padding: 8px;">{heure}</td>
            </tr>
        </table>
        <br>
        <a href="{dashboard_url}"
           style="background:#28a745; color:white; padding:12px 24px;
                  text-decoration:none; border-radius:5px; font-size:16px;">
            👉 Répondre au client
        </a>
    </body></html>
    """

    await envoyer_email_html(
        emails_destinataires,
        f"[HAAC] Assistance requise — +{sender_id}",
        html_body
    )


async def classifier_session_ia(contexte_conversation: str) -> tuple[int | None, str | None]:
    """Classifie une conversation IA complète dans une problématique."""
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Problematique).order_by(Problematique.libelle.asc())
            )
            problematiques = result.scalars().all()

        if not problematiques:
            return None, None

        libelles_map = {p.libelle: p.id for p in problematiques}

        prompt = f"""Tu es un assistant de classification pour la HAAC 
            (Haute Autorité de l'Audiovisuel et de la Communication du Bénin).

            Voici les catégories de problématiques officielles :
            {chr(10).join(f"- {lib}" for lib in libelles_map.keys())}

            Voici une conversation complète entre un citoyen et le chatbot HAAC :
            {contexte_conversation[:1500]}

            Dans quelle catégorie de problématique cette conversation s'inscrit-elle ?
            Réponds UNIQUEMENT avec le nom exact de la catégorie, rien d'autre."""

        loop = asyncio.get_event_loop()
        llm = chatbot["llm"]
        reponse_llm = await loop.run_in_executor(None, llm.invoke, prompt)

        libelle = reponse_llm.content.strip() \
            if hasattr(reponse_llm, 'content') else str(reponse_llm).strip()

        # Correspondance exacte
        if libelle in libelles_map:
            return libelles_map[libelle], libelle

        # Correspondance partielle
        for lib in libelles_map:
            if lib.lower() in libelle.lower() or libelle.lower() in lib.lower():
                return libelles_map[lib], lib

        # Fallback première problématique
        first = problematiques[0]
        return first.id, first.libelle

    except Exception as e:
        print(f"[CLASSIFIER] ❌ Erreur : {e}")
        return None, None


async def wait_and_send_followup(sender_id: str):
    """Attend puis envoie une relance. Si pas de réponse, clôture la session."""
    try:
        await asyncio.sleep(FOLLOWUP_DELAY)

        # Vérifier si conversation humaine active
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.phone == sender_id,
                    Conversation.statut.in_(["HUMAIN", "PRISE"])
                )
            )
            conv = result.scalar_one_or_none()

        if conv:
            return

        current_time = time.time()
        elapsed = current_time - last_message_time.get(sender_id, 0)

        if elapsed >= FOLLOWUP_DELAY:
            send_whatsapp_message(sender_id, FOLLOWUP_MESSAGE)
            print(f"[FOLLOWUP] 📤 Relance envoyée à {sender_id}")
            last_message_time[sender_id] = current_time

            # Lancer le timer de clôture par timeout
            if sender_id in SESSIONS_IA_ACTIVES:
                TIMEOUT_TASKS[sender_id] = asyncio.create_task(
                    attendre_et_cloturer_timeout(sender_id)
                )

    except asyncio.CancelledError:
        pass


async def attendre_et_cloturer_timeout(sender_id: str):
    """
    Attend TIMEOUT_CLOTURE secondes après la relance.
    Si le client ne répond toujours pas, envoie le message
    de clôture et clôture la session.
    """
    try:
        await asyncio.sleep(TIMEOUT_CLOTURE)

        # Vérifier qu'il n'y a pas eu de réponse entre-temps
        current_time = time.time()
        elapsed = current_time - last_message_time.get(sender_id, 0)

        if elapsed >= TIMEOUT_CLOTURE:
            # Envoyer le message de clôture
            send_whatsapp_message(sender_id, GOODBYE_MESSAGE)
            print(f"[TIMEOUT] 🔒 Session clôturée par timeout pour {sender_id}")

            # Clôturer la session IA
            await cloturer_session_ia(sender_id, "CLOTUREE_TIMEOUT")

    except asyncio.CancelledError:
        pass


def reset_followup_timer(sender_id: str):
    """Annule l'ancienne planification de relance et en crée une nouvelle."""
    last_message_time[sender_id] = time.time()
    if sender_id in followup_tasks and not followup_tasks[sender_id].done():
        followup_tasks[sender_id].cancel()
    followup_tasks[sender_id] = asyncio.create_task(wait_and_send_followup(sender_id))


async def cloturer_session_ia(
    sender_id: str,
    raison: str = "CLOTUREE_AU_REVOIR"
):
    """
    Clôture une session IA :
    1. Récupère tous les messages de la session
    2. Classifie via Gemini
    3. Enregistre en BD
    4. Nettoie les timers
    """
    session_id = SESSIONS_IA_ACTIVES.get(sender_id)
    if not session_id:
        return

    try:
        async with AsyncSessionLocal() as db:
            # Récupérer la session
            result = await db.execute(
                select(SessionIA).where(SessionIA.id == session_id)
            )
            session = result.scalar_one_or_none()
            if not session or session.statut != "EN_COURS":
                return

            # Récupérer tous les messages de la session
            result_msgs = await db.execute(
                select(MessageSessionIA)
                .where(MessageSessionIA.session_id == session_id)
                .order_by(MessageSessionIA.envoye_le.asc())
            )
            messages = result_msgs.scalars().all()

            # Classifier si on a des messages
            prob_id, prob_libelle = None, None
            if messages:
                # Construire le contexte de la conversation
                contexte = "\n".join([
                    f"{m.expediteur.upper()}: {m.texte}"
                    for m in messages
                ])
                prob_id, prob_libelle = await classifier_session_ia(contexte)

            # Mettre à jour la session
            session.statut         = raison
            session.cloturee_le    = datetime.utcnow()
            session.problematique_id      = prob_id
            session.problematique_libelle = prob_libelle
            await db.commit()

        print(f"[SESSION IA] ✅ Session {session_id} clôturée ({raison}) → {prob_libelle}")

    except Exception as e:
        print(f"[SESSION IA] ❌ Erreur clôture : {e}")
    finally:
        # Nettoyer la mémoire
        SESSIONS_IA_ACTIVES.pop(sender_id, None)
        if sender_id in TIMEOUT_TASKS:
            TIMEOUT_TASKS[sender_id].cancel()
            TIMEOUT_TASKS.pop(sender_id, None)


async def process_whatsapp_pipeline(sender_id: str, user_text: str):
    try:
        if is_rate_limited(sender_id):
            send_whatsapp_message(sender_id, SPAM_REPLY)
            return

        # Vérifier si conversation humaine active en BD
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.phone == sender_id,
                    Conversation.statut.in_(["HUMAIN", "PRISE"])
                )
            )
            conv = result.scalar_one_or_none()

        if conv:
            async with AsyncSessionLocal() as db:
                result_session = await db.execute(
                    select(Session).where(
                        Session.phone == sender_id,
                        Session.statut.in_(["HUMAIN", "PRISE"])
                    ).order_by(Session.cree_le.desc()).limit(1)
                )
                session_active = result_session.scalar_one_or_none()

                db.add(Message(
                    phone=sender_id,
                    session_id=session_active.id if session_active else None,
                    expediteur="client",
                    texte=user_text
                ))
                await db.commit()
            return

        # Vérifier si en attente de note client
        async with AsyncSessionLocal() as db:
            result_note = await db.execute(
                select(Session).where(
                    Session.phone == sender_id,
                    Session.en_attente_note == True
                ).order_by(Session.cloturee_le.desc()).limit(1)
            )
            session_note = result_note.scalar_one_or_none()

        note_text = user_text.strip().lower()
        if session_note:
            NOTE_MAP = {
                "1": 1, "très insatisfait": 1, "tres insatisfait": 1,
                "2": 2, "insatisfait": 2,
                "3": 3, "moyen": 3, "neutre": 3,
                "4": 4, "satisfait": 4,
                "5": 5, "très satisfait": 5, "tres satisfait": 5,
            }
            note_val = NOTE_MAP.get(note_text)
            if note_val:
                async with AsyncSessionLocal() as db:
                    session_note.note_client = note_val
                    session_note.en_attente_note = False
                    db.add(session_note)
                    await db.commit()
                send_whatsapp_message(sender_id,
                    "🙏 *Merci pour votre retour !*\n\n"
                    "Votre avis nous aide à améliorer nos services.")
                return
            else:
                send_whatsapp_message(sender_id,
                    "Merci de répondre avec :\n"
                    "• *Très insatisfait* • *Insatisfait*\n"
                    "• *Moyen* • *Satisfait* • *Très satisfait*")
                return

        cleaned_text = user_text.strip().lower().replace(".", "").replace("!", "")

        # Clôture par au revoir
        is_goodbye = (cleaned_text in GOODBYE_TRIGGERS) or \
                     check_if_goodbye_llm(chatbot["llm"], user_text)

        if is_goodbye:
            send_whatsapp_message(sender_id, GOODBYE_MESSAGE)

            # Clôturer la session IA si active
            if sender_id in SESSIONS_IA_ACTIVES:
                asyncio.create_task(
                    cloturer_session_ia(sender_id, "CLOTUREE_AU_REVOIR")
                )

            # Annuler les timers
            if sender_id in followup_tasks:
                followup_tasks[sender_id].cancel()
            if sender_id in TIMEOUT_TASKS:
                TIMEOUT_TASKS[sender_id].cancel()
            return

        # Messages triviaux
        history = chatbot["memory"].get_formatted_history(sender_id)
        trivial_response = handle_trivial(user_text, llm=chatbot["llm"], history=history)
        if trivial_response:
            send_whatsapp_message(sender_id, trivial_response)
            reset_followup_timer(sender_id)
            return

        # Pipeline RAG
        t0 = time.time()
        result = poser_question_avec_memoire(chatbot, user_text, user_id=sender_id)
        bot_answer_raw = result.get('response', '')

        # Handover
        if "[TRIGGER_HANDOVER]" in bot_answer_raw:
            # Clôturer session IA avant de passer en humain
            if sender_id in SESSIONS_IA_ACTIVES:
                asyncio.create_task(
                    cloturer_session_ia(sender_id, "CLOTUREE_AU_REVOIR")
                )

            async with AsyncSessionLocal() as db:
                # Créer la session humaine (celle que le dashboard agent affiche)
                nouvelle_session = Session(phone=sender_id, statut="HUMAIN")
                db.add(nouvelle_session)
                await db.flush()

                db.add(Message(
                    phone=sender_id,
                    session_id=nouvelle_session.id,
                    expediteur="client",
                    texte=user_text
                ))

                # Marquer/mettre à jour le pointeur de statut global par numéro
                conv_existante = await db.execute(
                    select(Conversation).where(Conversation.phone == sender_id)
                )
                conv_ex = conv_existante.scalar_one_or_none()
                if not conv_ex:
                    db.add(Conversation(phone=sender_id, statut="HUMAIN"))
                else:
                    conv_ex.statut = "HUMAIN"

                await db.commit()

            send_whatsapp_message(sender_id,
                "Veuillez patienter un instant, je vous mets en relation avec un agent de la HAAC 😊...")
            asyncio.create_task(notifier_agents_par_email(sender_id, user_text))

            if sender_id in followup_tasks:
                followup_tasks[sender_id].cancel()
            return

        # Réponse normale de l'IA
        bot_answer = markdown_to_whatsapp(bot_answer_raw)
        send_whatsapp_message(sender_id, bot_answer)

        # Créer ou mettre à jour la session IA active
        if sender_id not in SESSIONS_IA_ACTIVES:
            async with AsyncSessionLocal() as db:
                nouvelle_session = SessionIA(phone=sender_id, statut="EN_COURS")
                db.add(nouvelle_session)
                await db.flush()
                session_id = nouvelle_session.id

                # Ajouter les deux messages (question + réponse)
                db.add(MessageSessionIA(
                    session_id=session_id,
                    phone=sender_id,
                    expediteur="client",
                    texte=user_text
                ))
                db.add(MessageSessionIA(
                    session_id=session_id,
                    phone=sender_id,
                    expediteur="ia",
                    texte=bot_answer_raw,
                    duree_ms=int((time.time() - t0) * 1000)
                ))
                await db.commit()

            SESSIONS_IA_ACTIVES[sender_id] = session_id
        else:
            # Ajouter à la session existante
            session_id = SESSIONS_IA_ACTIVES[sender_id]
            async with AsyncSessionLocal() as db:
                db.add(MessageSessionIA(
                    session_id=session_id,
                    phone=sender_id,
                    expediteur="client",
                    texte=user_text
                ))
                db.add(MessageSessionIA(
                    session_id=session_id,
                    phone=sender_id,
                    expediteur="ia",
                    texte=bot_answer_raw,
                    duree_ms=int((time.time() - t0) * 1000)
                ))

                # Incrémenter le compteur d'échanges
                await db.execute(
                    update(SessionIA)
                    .where(SessionIA.id == session_id)
                    .values(nb_echanges=SessionIA.nb_echanges + 1)
                )
                await db.commit()

        # Gérer les timers de relance et timeout
        reset_followup_timer(sender_id)

    except Exception as e:
        print(f"[PIPELINE-ERREUR] ❌ Dysfonctionnement : {e}")
        send_whatsapp_message(sender_id,
            "Navré, mes systèmes rencontrent une surcharge temporaire. "
            "Veuillez reformuler.")


# =============================================================================================


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Vérifie le webhook WhatsApp lors du handshake Meta."""
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        print("[WEBHOOK] ✅ Handshake Meta validé")
        return Response(content=params.get("hub.challenge"), status_code=200)
    return Response(content="Verification failed", status_code=403)


@app.post("/webhook")
async def handle_message(request: Request):
    """Reçoit les messages WhatsApp entrants envoyés par Meta."""
    data = await request.json()
    immediate_response = JSONResponse(content={"status": "accepted"}, status_code=200)

    try:
        entry = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {})
        if not entry or 'messages' not in entry:
            return immediate_response

        message = entry['messages'][0]
        if message.get('type') != 'text':
            return immediate_response

        message_id = message.get('id', '')
        if message_id in processed_message_ids:
            return immediate_response

        processed_message_ids.add(message_id)
        if len(processed_message_ids) > MAX_PROCESSED_IDS:
            processed_message_ids.pop()

        sender_id = message['from']
        user_text = message['text']['body']

        print(f"\n[WEBHOOK] 📩 Nouveau message de : {sender_id} (ID: {message_id})")
        asyncio.create_task(process_whatsapp_pipeline(sender_id, user_text))

    except Exception as e:
        print(f"[WEBHOOK-ERREUR] ❌ Erreur critique parsing : {e}")

    return immediate_response


@app.post("/api/ask")
async def ask_question(request_data: QuestionRequest):
    """Teste le chatbot HAAC directement sans passer par WhatsApp."""
    user_id = request_data.user_id or f"web_user_{int(time.time())}"
    result = poser_question_avec_memoire(chatbot, request_data.question, user_id=user_id)
    return {"response": result['response'], "sources": result['sources']}


embeddings = HuggingFaceAPIEmbeddings(api_key=HF_TOKEN)


@app.get("/conversations-humaines")
async def get_human_conversations(
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Retourne toutes les sessions avec leurs messages associés."""
    result = await db.execute(
        select(Session).order_by(Session.cree_le.asc())
    )
    sessions = result.scalars().all()

    data = []
    for s in sessions:
        msgs_result = await db.execute(
            select(Message)
            .where(Message.session_id == s.id)
            .order_by(Message.envoye_le.asc())
        )
        messages = msgs_result.scalars().all()
        data.append({
            "id": s.id,
            "phone": s.phone,
            "statut": s.statut,
            "agent": s.agent,
            "agent_cloture": s.agent_cloture,
            "date_prise_charge": s.prise_en_charge_le.strftime("%d/%m/%Y à %H:%M") if s.prise_en_charge_le else None,
            "date_cloture": s.cloturee_le.strftime("%d/%m/%Y à %H:%M") if s.cloturee_le else None,
            "en_attente_depuis": s.cree_le.strftime("%d/%m/%Y à %H:%M") if s.cree_le else "inconnue",
            "messages": [
                {
                    "expediteur": m.expediteur,
                    "text": m.texte,
                    "timestamp": m.envoye_le.strftime("%H:%M") if m.envoye_le else "",
                    "nom_agent": m.nom_agent,
                }
                for m in messages
            ]
        })
    return {"conversations": data}


@app.post("/prendre-en-charge/{session_id}")
async def prendre_conversation(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Permet à un agent de prendre en charge une session en attente."""
    s = await db.get(Session, session_id)

    if not s or s.statut != "HUMAIN":
        raise HTTPException(status_code=400, detail="Session non disponible.")

    s.statut = "PRISE"
    s.agent = agent.nom
    s.prise_en_charge_le = datetime.utcnow()
    await db.commit()

    return {"status": "success", "message": f"Session assignée à {agent.nom}."}


@app.post("/repondre/{session_id}")
async def agent_reply(
    session_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Envoie un message WhatsApp au client depuis l'agent."""
    s = await db.get(Session, session_id)

    if not s or s.statut not in ("HUMAIN", "PRISE"):
        raise HTTPException(status_code=400, detail="Session non disponible.")

    body = await request.json()
    message = body.get("message", "").strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message vide.")

    if not s.premiere_reponse_le:
        s.premiere_reponse_le = datetime.utcnow()

    send_whatsapp_message(s.phone, message)
    db.add(Message(
        phone=s.phone,
        session_id=s.id,
        expediteur="agent",
        texte=message,
        nom_agent=agent.nom
    ))
    await db.commit()

    return {"status": "success", "message": "Réponse envoyée."}


@app.get("/problematiques")
async def lister_problematiques_public(
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Liste les problématiques disponibles pour le formulaire de clôture."""
    result = await db.execute(select(Problematique).order_by(Problematique.libelle.asc()))
    items = result.scalars().all()
    return [{"id": p.id, "libelle": p.libelle} for p in items]


@app.post("/cloturer-session/{session_id}")
async def close_human_session(
    session_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Clôture une session humaine et demande une note de satisfaction."""
    s = await db.get(Session, session_id)

    if not s or s.statut not in ("HUMAIN", "PRISE"):
        return {"status": "error", "message": "Session non trouvée."}

    body = await request.json()
    s.problematique = body.get("problematique", None)
    s.commentaire_agent = body.get("commentaire_agent", None)
    s.agent_cloture = s.agent
    s.cloturee_le = datetime.utcnow()
    s.statut = "CLOTUREE"
    s.agent = None
    s.en_attente_note = True

    # Libérer le numéro (le chatbot IA peut reprendre la main)
    result_conv = await db.execute(
        select(Conversation).where(Conversation.phone == s.phone)
    )
    conv = result_conv.scalar_one_or_none()
    if conv:
        conv.statut = "IA"

    await db.commit()

    send_whatsapp_message(s.phone, GOODBYE_MESSAGE)
    send_whatsapp_message(s.phone, (
        "⭐ *Votre avis compte !*\n\n"
        "Comment évaluez-vous la qualité de l'assistance reçue ?\n\n"
        "Répondez avec un chiffre de 1 à 5 :\n"
        "1 — ⭐ Très insatisfait\n2 — ⭐⭐ Insatisfait\n3 — ⭐⭐⭐ Moyen\n4 — ⭐⭐⭐⭐ Satisfait\n5 — ⭐⭐⭐⭐⭐ Très satisfait"
    ))

    print(f"[HANDOVER] 🚪 Session {session_id} close pour {s.phone} par {agent.nom}.")
    return {"status": "success", "message": "Session clôturée."}


@app.post("/auth/login")
async def login(request: Request, db: AsyncSession = Depends(get_db)):
    """Authentifie un agent ou un admin."""
    body = await request.json()
    email = body.get("email", "").strip()
    mot_de_passe = body.get("mot_de_passe", "").strip()

    result = await db.execute(select(Agent).where(Agent.email == email))
    agent = result.scalar_one_or_none()

    if not agent or not verifier_mot_de_passe(mot_de_passe, agent.mot_de_passe):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect.")

    if not agent.actif:
        raise HTTPException(status_code=403, detail="Compte désactivé. Contactez l'administrateur.")

    token = creer_token({"sub": agent.email, "role": agent.role, "nom": agent.nom})
    return {
        "access_token": token,
        "token_type": "bearer",
        "nom": agent.nom,
        "role": agent.role
    }


@app.get("/admin/dashboard")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Retourne les statistiques globales en temps réel."""
    aujourd_hui = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    r1 = await db.execute(select(func.count(Session.id)).where(Session.statut == "HUMAIN"))
    en_attente = r1.scalar() or 0

    r2 = await db.execute(select(func.count(Session.id)).where(Session.statut == "PRISE"))
    en_cours = r2.scalar() or 0

    r3 = await db.execute(
        select(func.count(Session.id))
        .where(Session.cloturee_le >= aujourd_hui, Session.statut == "CLOTUREE")
    )
    cloturees_jour = r3.scalar() or 0

    r4 = await db.execute(select(func.count(Session.id)).where(Session.statut == "CLOTUREE"))
    cloturees_total = r4.scalar() or 0

    r5 = await db.execute(select(func.count(Session.id)))
    total_conversations = r5.scalar() or 0

    r6 = await db.execute(select(func.count(Agent.id)).where(Agent.actif == True))
    agents_actifs = r6.scalar() or 0

    r7 = await db.execute(select(func.count(Agent.id)))
    agents_total = r7.scalar() or 0

    r8 = await db.execute(
        select(func.count(Message.id)).where(Message.envoye_le >= aujourd_hui)
    )
    messages_jour = r8.scalar() or 0

    return {
        "en_attente": en_attente,
        "en_cours": en_cours,
        "cloturees_jour": cloturees_jour,
        "cloturees_total": cloturees_total,
        "total_conversations": total_conversations,
        "agents_actifs": agents_actifs,
        "agents_total": agents_total,
        "messages_jour": messages_jour
    }


@app.post("/admin/creer-agent")
async def creer_agent(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Crée un nouveau compte agent ou admin et envoie un email d’invitation."""
    body = await request.json()
    nom = body.get("nom", "").strip()
    email = body.get("email", "").strip()
    mot_de_passe = body.get("mot_de_passe", "").strip()
    role = body.get("role", "agent").strip()

    if not nom or not email:
        raise HTTPException(status_code=400, detail="Nom et email sont obligatoires.")

    if role not in ("agent", "admin"):
        raise HTTPException(status_code=400, detail="Le rôle doit être 'agent' ou 'admin'.")

    result = await db.execute(select(Agent).where(Agent.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Un compte avec cet email existe déjà.")

    if not mot_de_passe:
        mot_de_passe = f"Haac2026!{email.split('@')[0][:4].upper()}"

    nouvel_agent = Agent(
        nom=nom,
        email=email,
        mot_de_passe=hasher_mot_de_passe(mot_de_passe),
        role=role,
        actif=True
    )
    db.add(nouvel_agent)
    await db.commit()

    html_body = f"""
    <html><body style="font-family: Arial, sans-serif; color: #333;">
        <h2 style="color: #0b66c2;">Bienvenue sur la plateforme HAAC</h2>
        <p>Bonjour {nom},</p>
        <p>Votre compte a été créé sur la plateforme HAAC.</p>
        <p><strong>Email :</strong> {email}</p>
        <p><strong>Mot de passe temporaire :</strong> {mot_de_passe}</p>
        <p>Veuillez vous connecter puis modifier votre mot de passe dès votre première connexion.</p>
        <p>Merci,</p>
        <p>L'équipe HAAC</p>
    </body></html>
    """

    background_tasks.add_task(envoyer_email_html, [email], "Votre accès à la plateforme HAAC", html_body)

    print(f"[ADMIN] ✅ Compte {role} créé : {email} par {admin.nom}")
    return {
        "status": "success",
        "message": f"Compte {role} créé pour {nom}.",
        "mot_de_passe_temporaire": mot_de_passe
    }


@app.get("/admin/agents")
async def lister_agents(
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Liste tous les agents et admins."""
    result = await db.execute(select(Agent).order_by(Agent.cree_le.desc()))
    agents = result.scalars().all()
    return [
        {
            "id": a.id,
            "nom": a.nom,
            "email": a.email,
            "role": a.role,
            "actif": a.actif,
            "cree_le": a.cree_le.strftime("%d/%m/%Y") if a.cree_le else ""
        }
        for a in agents
    ]


async def modifier_agent(
    agent_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Modifie le rôle et/ou le statut (actif) d'un agent.
    L'admin ne peut modifier que le rôle et le statut des autres comptes,
    jamais leur nom, email ou mot de passe."""
    body = await request.json()
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent introuvable.")

    if "role" in body and body["role"] in ("agent", "admin", "superadmin"):
        agent.role = body["role"]
    if "actif" in body:
        agent.actif = body["actif"]

    await db.commit()
    return {"status": "success", "message": "Agent mis à jour."}


@app.get("/admin/me")
async def get_admin_profile(
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Retourne les informations du profil admin connecté."""
    return {
        "id": admin.id,
        "nom": admin.nom,
        "email": admin.email,
        "role": admin.role,
        "actif": admin.actif,
        "cree_le": admin.cree_le.strftime("%d/%m/%Y à %H:%M") if admin.cree_le else None,
        "mot_de_passe": "***"
    }


@app.put("/admin/me")
async def update_admin_profile(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Permet à l'admin connecté de modifier son nom et son email.
    Le mot de passe se change uniquement via PUT /me/password."""
    body = await request.json()

    if "nom" in body and body["nom"].strip():
        admin.nom = body["nom"].strip()
    if "email" in body and body["email"].strip():
        existing = await db.execute(select(Agent).where(Agent.email == body["email"].strip(), Agent.id != admin.id))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Cet email est déjà utilisé.")
        admin.email = body["email"].strip()

    await db.commit()
    return {"status": "success", "message": "Profil admin mis à jour."}


@app.post("/admin/logout")
async def logout_admin():
    """Déconnexion côté API pour l'admin."""
    return {"status": "success", "message": "Déconnecté."}


@app.get("/agent/me")
async def get_agent_profile(
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Retourne les informations du profil agent connecté."""
    return {
        "id": agent.id,
        "nom": agent.nom,
        "email": agent.email,
        "role": agent.role,
        "actif": agent.actif,
        "cree_le": agent.cree_le.strftime("%d/%m/%Y à %H:%M") if agent.cree_le else None,
        "mot_de_passe": "***"
    }


@app.put("/agent/me")
async def update_agent_profile(
    request: Request,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(get_agent_connecte)
):
    """Permet à l'agent connecté de modifier son nom et son email.
    Le mot de passe se change uniquement via PUT /me/password."""
    body = await request.json()

    if "nom" in body and body["nom"].strip():
        agent.nom = body["nom"].strip()
    if "email" in body and body["email"].strip():
        existing = await db.execute(select(Agent).where(Agent.email == body["email"].strip(), Agent.id != agent.id))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Cet email est déjà utilisé.")
        agent.email = body["email"].strip()

    await db.commit()
    return {"status": "success", "message": "Profil agent mis à jour."}


@app.post("/agent/logout")
async def logout_agent():
    """Déconnexion côté API pour l'agent."""
    return {"status": "success", "message": "Déconnecté."}


@app.put("/me/password")
async def changer_mot_de_passe_compte(
    request: Request,
    db: AsyncSession = Depends(get_db),
    compte: Agent = Depends(get_agent_connecte)
):
    """Permet à tout compte connecté (agent ou admin) de changer son mot de passe."""
    body = await request.json()
    mot_de_passe_actuel = body.get("mot_de_passe_actuel", "").strip()
    nouveau_mot_de_passe = body.get("nouveau_mot_de_passe", "").strip()

    if not mot_de_passe_actuel or not nouveau_mot_de_passe:
        raise HTTPException(status_code=400, detail="Les deux mots de passe sont obligatoires.")

    if not verifier_mot_de_passe(mot_de_passe_actuel, compte.mot_de_passe):
        raise HTTPException(status_code=401, detail="Mot de passe actuel incorrect.")

    if len(nouveau_mot_de_passe) < 6:
        raise HTTPException(status_code=400, detail="Le nouveau mot de passe doit contenir au moins 6 caractères.")

    compte.mot_de_passe = hasher_mot_de_passe(nouveau_mot_de_passe)
    await db.commit()

    return {"status": "success", "message": "Mot de passe mis à jour."}


@app.delete("/admin/agents/{agent_id}")
async def desactiver_agent(
    agent_id: int,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Désactive un compte agent (soft delete)."""
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalar_one_or_none()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent introuvable.")

    if agent.role == "admin":
        result2 = await db.execute(
            select(Agent).where(Agent.role == "admin", Agent.actif == True)
        )
        admins_actifs = result2.scalars().all()
        if len(admins_actifs) <= 1:
            raise HTTPException(
                status_code=400,
                detail="Impossible de désactiver le dernier admin actif."
            )

    agent.actif = False
    await db.commit()
    return {"status": "success", "message": f"Compte de {agent.nom} désactivé."}


@app.get("/admin/statistiques")
async def get_statistiques(
    periode: str = "semaine",
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Retourne des statistiques détaillées filtrées par période."""
    maintenant = datetime.utcnow()
    if periode == "aujourd'hui":
        tronc = func.date_trunc('hour', Conversation.cree_le).label("jour")
        format_str = "%Hh"
    elif periode in ("semaine", "mois"):
        tronc = func.date_trunc('day', Conversation.cree_le).label("jour")
        format_str = "%d/%m"
    else:  # "tout"
        tronc = func.date_trunc('month', Conversation.cree_le).label("jour")
        format_str = "%m/%Y"

    if periode == "aujourd'hui":
        date_debut = maintenant.replace(hour=0, minute=0, second=0, microsecond=0)
    elif periode == "semaine":
        date_debut = maintenant - timedelta(days=7)
    elif periode == "mois":
        date_debut = maintenant - timedelta(days=30)
    else:  # "tout"
        date_debut = datetime(2000, 1, 1)

    filtre_periode = Session.cree_le >= date_debut

    result_par_jour = await db.execute(
        select(
            tronc,
            func.count(Conversation.phone).label("total")
        )
        .where(filtre_periode)
        .group_by(tronc)
        .order_by(tronc)
    )
    par_jour = result_par_jour.all()

    
    r_total = await db.execute(select(func.count(Session.id)).where(filtre_periode))
    total_conversations = r_total.scalar() or 0

    r_cloturees = await db.execute(
        select(func.count(Session.id))
        .where(and_(filtre_periode, Session.cloturee_le != None))
    )
    total_cloturees = r_cloturees.scalar() or 0

    r_statuts = await db.execute(
        select(Session.statut, func.count(Session.id).label("total"))
        .group_by(Session.statut)
    )
    statuts = {row.statut: row.total for row in r_statuts.all()}

    r_agents = await db.execute(
        select(
            Session.agent_cloture,
            func.count(Session.id).label("total_conversations"),
            func.avg(
                func.extract('epoch', Session.prise_en_charge_le) -
                func.extract('epoch', Session.cree_le)
            ).label("temps_attente_moy_sec"),
            func.avg(
                func.extract('epoch', Session.premiere_reponse_le) -
                func.extract('epoch', Session.prise_en_charge_le)
            ).label("temps_reponse_moy_sec"),
            func.avg(Session.note_client).label("note_client_moy"),
            func.count(Session.note_client).label("nb_notes")
        )
        .where(and_(filtre_periode, Session.agent_cloture != None))
        .group_by(Session.agent_cloture)
        .order_by(func.count(Session.id).desc())
    )
    stats_agents = r_agents.all()

    r_note = await db.execute(
        select(func.avg(Session.note_client))
        .where(and_(filtre_periode, Session.note_client != None))
    )
    note_globale = r_note.scalar()

    r_prob = await db.execute(
        select(Session.problematique, func.count(Session.id).label("total"))
        .where(and_(filtre_periode, Session.problematique != None))
        .group_by(Session.problematique)
        .order_by(func.count(Session.id).desc())
    )
    problematiques = r_prob.all()

    r_par_jour = await db.execute(
        sa_text("""
            SELECT date_trunc('day', cree_le) AS jour, count(id) AS total
            FROM sessions
            WHERE cree_le >= :date_debut
            GROUP BY date_trunc('day', cree_le)
            ORDER BY date_trunc('day', cree_le)
        """),
        {"date_debut": date_debut}
    )
    par_jour = r_par_jour.all()

    r_notes = await db.execute(
        select(Session.note_client, func.count(Session.id).label("total"))
        .where(and_(filtre_periode, Session.note_client != None))
        .group_by(Session.note_client)
        .order_by(Session.note_client.desc())
    )
    notes_raw = r_notes.all()
    total_notes = sum(r.total for r in notes_raw)
    repartition_notes = [
        {
            "note": r.note_client,
            "total": r.total,
            "pourcentage": round((r.total / total_notes) * 100, 1) if total_notes > 0 else 0
        }
        for r in notes_raw
    ]

    return {
        "periode": periode,
        "resume": {
            "total_conversations": total_conversations,
            "total_cloturees": total_cloturees,
            "en_attente": statuts.get("HUMAIN", 0),
            "en_cours": statuts.get("PRISE", 0),
            "note_client_globale": round(note_globale, 1) if note_globale else None
        },
        "stats_agents": [
            {
                "agent": r.agent_cloture,
                "total_conversations": r.total_conversations,
                "temps_attente_moy_min": round(r.temps_attente_moy_sec / 60, 1) if r.temps_attente_moy_sec else None,
                "temps_reponse_moy_min": round(r.temps_reponse_moy_sec / 60, 1) if r.temps_reponse_moy_sec else None,
                "note_client_moy": round(r.note_client_moy, 1) if r.note_client_moy else None,
                "nb_notes": r.nb_notes,
                "etoiles": round(r.note_client_moy) if r.note_client_moy else None
            }
            for r in stats_agents
        ],
        "problematiques": [{"label": r.problematique, "total": r.total} for r in problematiques],
        "conversations_par_jour": [{"jour": r.jour.strftime("%d/%m"), "total": r.total} for r in par_jour],
        "repartition_notes": repartition_notes,
        "conversations_par_jour": [
            {
                "jour": r.jour.strftime(format_str),
                "total": r.total
            }
            for r in par_jour
        ]

    }


@app.get("/admin/stats-ia")
async def get_stats_ia(
    periode: str = "semaine",
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    maintenant = datetime.utcnow()
    if periode == "aujourd'hui":
        date_debut = maintenant.replace(hour=0, minute=0, second=0, microsecond=0)
        tronc = func.date_trunc('hour', SessionIA.debut_le).label("periode")
        format_str = "%Hh"
    elif periode == "semaine":
        date_debut = maintenant - timedelta(days=7)
        tronc = func.date_trunc('day', SessionIA.debut_le).label("periode")
        format_str = "%d/%m"
    elif periode == "mois":
        date_debut = maintenant - timedelta(days=30)
        tronc = func.date_trunc('day', SessionIA.debut_le).label("periode")
        format_str = "%d/%m"
    else:
        date_debut = datetime(2000, 1, 1)
        tronc = func.date_trunc('month', SessionIA.debut_le).label("periode")
        format_str = "%m/%Y"

    filtre = and_(SessionIA.debut_le >= date_debut, SessionIA.statut != "EN_COURS")

    # Total sessions IA clôturées
    r1 = await db.execute(select(func.count(SessionIA.id)).where(filtre))
    total_sessions = r1.scalar() or 0

    # Sessions clôturées vers handover
    r2 = await db.execute(
        select(func.count(Session.id))
        .where(Session.cree_le >= date_debut)
    )
    total_handovers = r2.scalar() or 0

    # Taux de handover
    total_general = total_sessions + total_handovers
    taux_handover = round((total_handovers / total_general * 100), 1) \
        if total_general > 0 else 0

    # Temps de réponse moyen (depuis MessageSessionIA)
    r3 = await db.execute(
        select(func.avg(MessageSessionIA.duree_ms))
        .join(SessionIA, MessageSessionIA.session_id == SessionIA.id)
        .where(
            and_(
                SessionIA.debut_le >= date_debut,
                MessageSessionIA.expediteur == "ia",
                MessageSessionIA.duree_ms != None
            )
        )
    )
    duree_moy = r3.scalar()

    # Sessions par période
    r4 = await db.execute(
        select(tronc, func.count(SessionIA.id).label("total"))
        .where(filtre)
        .group_by(tronc)
        .order_by(tronc)
    )
    par_periode = r4.all()

    return {
        "periode": periode,
        "total_sessions_ia": total_sessions,
        "total_handovers": total_handovers,
        "taux_handover_pct": taux_handover,
        "duree_reponse_moy_sec": round(duree_moy / 1000, 2) if duree_moy else None,
        "sessions_par_periode": [
            {"jour": r.periode.strftime(format_str), "total": r.total}
            for r in par_periode
        ]
    }


@app.get("/admin/questions-frequentes")
async def get_questions_frequentes(
    limite: int = 10,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Retourne les dernières questions posées au chatbot IA."""
    result = await db.execute(
        select(MessageSessionIA)
        .where(MessageSessionIA.expediteur == "ia")
        .order_by(MessageSessionIA.envoye_le.desc())
        .limit(limite)
    )
    reponses = result.scalars().all()

    questions = []
    for r in reponses:
        result_q = await db.execute(
            select(MessageSessionIA)
            .where(
                MessageSessionIA.session_id == r.session_id,
                MessageSessionIA.expediteur == "client",
                MessageSessionIA.envoye_le <= r.envoye_le,
            )
            .order_by(MessageSessionIA.envoye_le.desc())
            .limit(1)
        )
        q = result_q.scalar_one_or_none()
        questions.append({
            "phone": r.phone,
            "question": q.texte if q else "(question introuvable)",
            "duree_sec": round(r.duree_ms / 1000, 2) if r.duree_ms else None,
            "date": r.envoye_le.strftime("%d/%m/%Y à %H:%M") if r.envoye_le else ""
        })

    return {"questions": questions}

@app.get("/admin/themes-frequents")
async def get_themes_frequents(
    periode: str = "semaine",
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    maintenant = datetime.utcnow()
    if periode == "aujourd'hui":
        date_debut = maintenant.replace(hour=0, minute=0, second=0, microsecond=0)
    elif periode == "semaine":
        date_debut = maintenant - timedelta(days=7)
    elif periode == "mois":
        date_debut = maintenant - timedelta(days=30)
    else:
        date_debut = datetime(2000, 1, 1)

    # Toutes les problématiques
    result_prob = await db.execute(
        select(Problematique).order_by(Problematique.libelle.asc())
    )
    problematiques = result_prob.scalars().all()

    # Compter les sessions IA clôturées par problématique
    result_counts = await db.execute(
        select(
            SessionIA.problematique_libelle,
            func.count(SessionIA.id).label("nombre")
        )
        .where(
            and_(
                SessionIA.cloturee_le >= date_debut,
                SessionIA.statut != "EN_COURS",
                SessionIA.problematique_libelle != None
            )
        )
        .group_by(SessionIA.problematique_libelle)
    )
    counts = {r.problematique_libelle: r.nombre for r in result_counts.all()}

    themes = [
        {"theme": p.libelle, "nombre": counts.get(p.libelle, 0)}
        for p in problematiques
    ]
    themes.sort(key=lambda x: x["nombre"], reverse=True)

    result_total = await db.execute(
        select(func.count(SessionIA.id))
        .where(
            and_(
                SessionIA.debut_le >= date_debut,
                SessionIA.statut != "EN_COURS"
            )
        )
    )
    total = result_total.scalar() or 0

    return {"themes": themes, "total_sessions_analysees": total}



@app.get("/admin/problematiques")
async def lister_problematiques(
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Liste toutes les problématiques (admin uniquement)."""
    result = await db.execute(select(Problematique).order_by(Problematique.libelle.asc()))
    items = result.scalars().all()
    return [{"id": p.id, "libelle": p.libelle} for p in items]


@app.post("/admin/problematiques")
async def creer_problematique(
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Crée une nouvelle problématique de clôture."""
    body = await request.json()
    libelle = body.get("libelle", "").strip()

    if not libelle:
        raise HTTPException(status_code=400, detail="Le libellé est obligatoire.")

    result = await db.execute(select(Problematique).where(Problematique.libelle == libelle))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Cette problématique existe déjà.")

    db.add(Problematique(libelle=libelle))
    await db.commit()
    return {"status": "success", "message": "Problématique créée."}


@app.delete("/admin/problematiques/{problematique_id}")
async def supprimer_problematique(
    problematique_id: int,
    db: AsyncSession = Depends(get_db),
    admin: Agent = Depends(get_admin_connecte)
):
    """Supprime définitivement une problématique."""
    result = await db.execute(select(Problematique).where(Problematique.id == problematique_id))
    prob = result.scalar_one_or_none()

    if not prob:
        raise HTTPException(status_code=404, detail="Problématique introuvable.")

    await db.delete(prob)
    await db.commit()
    return {"status": "success", "message": "Problématique supprimée."}


@app.post("/upload-docx/")
async def upload_docx_to_faiss(file: UploadFile = File(...)):
    """Convertit un fichier Word (.docx) en Markdown et l'indexe dans FAISS."""
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers .docx sont acceptés.")

    os.makedirs("temp_uploads", exist_ok=True)
    os.makedirs("markdowns", exist_ok=True)

    temp_path = f"temp_uploads/{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        style_map = """
        p[style-name='Heading 1'] => h1:id=false
        p[style-name='Heading 2'] => h2:id=false
        p[style-name='Heading 3'] => h3:id=false
        p[style-name='Title'] => h1:id=false
        """

        with open(temp_path, "rb") as docx_file:
            result = mammoth.convert_to_markdown(docx_file, style_map=style_map)
            markdown_content = result.value

        if not markdown_content.strip():
            raise HTTPException(status_code=400, detail="Le document Word est vide.")

        nom_fichier_md = file.filename.replace(".docx", ".md")
        chemin_sauvegarde_md = os.path.join("markdowns", nom_fichier_md)

        with open(chemin_sauvegarde_md, "w", encoding="utf-8") as f_md:
            f_md.write(markdown_content)
        print(f"💾 Fichier Markdown sauvegardé sous : {chemin_sauvegarde_md}")

        headers_to_split_on = [
            ("#", "Grand_Titre"),
            ("##", "Sous_Titre"),
            ("###", "Section_Article"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        sections = header_splitter.split_text(markdown_content)
        final_chunks = []

        for doc in sections:
            doc.metadata["source"] = nom_fichier_md
            sub_chunks = text_splitter.split_documents([doc])
            final_chunks.extend(sub_chunks)

        if os.path.exists(PATH_FAISS):
            vectorstore = FAISS.load_local(PATH_FAISS, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(final_chunks)
        else:
            vectorstore = FAISS.from_documents(final_chunks, embeddings)

        vectorstore.save_local(PATH_FAISS)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Le document a été converti, sauvegardé dans 'markdowns' et ajouté à FAISS !",
                "fichier_md_cree": nom_fichier_md,
                "chunks_ajoutes": len(final_chunks)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)