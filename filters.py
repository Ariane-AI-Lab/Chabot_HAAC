import re
import time
from collections import defaultdict
from retrieve import quota_is_exhausted

# ---------------------------------------------------------------------------
# CONSTANTES SPAM
# ---------------------------------------------------------------------------
SPAM_MAX_MESSAGES = 5
SPAM_WINDOW_SECONDS = 60

SPAM_REPLY = (
    "⚠️ Vous envoyez des messages trop rapidement. "
    "Merci de patienter quelques instants avant de poser votre prochaine question."
)

# ---------------------------------------------------------------------------
# CONFIGURATIONS ET MESSAGES DU CALL CENTER HAAC
# ---------------------------------------------------------------------------
FOLLOWUP_DELAY = 600  # 10 minutes en secondes

FOLLOWUP_MESSAGE = (
    "Hello! Je remarque que vous n'avez pas envoyé de message depuis un moment. "
    "Avez-vous d'autres préoccupations concernant la réglementation des médias sur lesquelles je peux vous aider ? 😊"
)

GOODBYE_TRIGGERS = ["non", "merci", "au revoir", "bye", "stop", "no", "nothing"]

GOODBYE_MESSAGE = (
    "C'est un plaisir de vous avoir assisté ! La HAAC vous remercie pour votre confiance. 🙏\n\n"
    "Nous restons à votre entière disposition pour toute autre préoccupation.\n"
    "Pour plus d'informations, visitez notre portail ou écrivez-nous :\n"
    "📧 contact@haac.bj\n"
    "🌐 https://haac.bj\n\n"
    "L'équipe d'assistance HAAC vous souhaite une excellente journée ! 😊"
)

# ---------------------------------------------------------------------------
# CONSTANTES MESSAGES TRIVIAUX
# ---------------------------------------------------------------------------
GREETING_PATTERNS = [
    r"^\s*(bonjour|bonsoir|salut|hello|hi|hey|coucou|allo|allô)\s*[!?.]*\s*$",
]
GREETING_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in GREETING_PATTERNS]

GREETING_REPLY = (
    "👋 Salut ! Je suis l'assistant officiel de la HAAC "
    "(Haute Autorité de l'Audiovisuel et de la Communication du Bénin).\n\n"
    "Je suis ici pour répondre à vos questions sur la réglementation audiovisuelle, "
    "les textes officiels et les procédures de la HAAC.\n\n"
    "En quoi puis-je vous aider ?"
)

TRIVIAL_PATTERNS = [
    r"^\s*(merci|thanks|thx|thank you)\s*[!?.]*\s*$",
    r"^\s*(ok|okay|👍|👎|😊|🙏)\s*$",
    r"^\s*.{0,2}\s*$",
    r"^\s*(test|testing|essai|123|ping)\s*[!?.]*\s*$",
]
TRIVIAL_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRIVIAL_PATTERNS]

# 📝 PROMPTS LLM (Trivia & Clôture)
TRIVIAL_RESPONSE_PROMPT = """Tu es l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication du Bénin).

Un utilisateur t'envoie ce message : "{text}"

Génère une réponse courte (1 à 2 phrases maximum), naturelle, chaleureuse et adaptée au contexte exact du message.
Ne te présente pas sauf si c'est une salutation d'ouverture. Réponds juste de manière humaine et cohérente.

Réponds UNIQUEMENT avec la réponse courte, rien d'autre."""

TRIVIAL_CLASSIFY_PROMPT = """Tu es l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication du Bénin).

Tu devez analyser le message actuel de l'utilisateur en prenant en compte l'historique de la conversation.

HISTORIQUE DE LA CONVERSATION :
{history}

MESSAGE ACTUEL DE L'UTILISATEUR : "{text}"

Ce message contient-il une vraie question, une demande d'information, ou une RÉPONSE à une question précédente du bot (comme préciser sa situation : "première demande", "renouvellement") ?
- Si OUI (c'est une question ou une réponse contextuelle importante) → réponds UNIQUEMENT avec le mot : QUESTION
- Si NON (c'est une phrase inutile, un hors-sujet total ou une banalité) → génère une réponse courte (1 à 2 phrases max) naturelle et adaptée.

Réponds UNIQUEMENT avec QUESTION ou avec la réponse courte, rien d'autre."""

CLOTURE_CLASSIFY_PROMPT = """Analyse le message court d'un utilisateur de chatbot et détermine s'il exprime la fin de la discussion (intention de dire au revoir, de remercier pour clore, ou d'indiquer qu'il n'a plus de questions).

MESSAGE DE L'UTILISATEUR : "{text}"

Réponds UNIQUEMENT par le mot OUI si l'utilisateur veut clore la discussion.
Réponds UNIQUEMENT par le mot NON si l'utilisateur pose une question ou attend une suite.

Exemples de OUI : "non c'est bon merci", "c'est tout pour moi", "merci bien", "fin", "plus de questions", "merci bonsoir".
Exemples de NON : "non, je veux plutôt savoir...", "merci mais qu'en est-il de...", "c'est bon pour la carte, et pour la radio ?".

Réponse (OUI ou NON) :"""

# ---------------------------------------------------------------------------
# ANTI-SPAM
# ---------------------------------------------------------------------------
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
# ANALYSE ET FILTRES DES MESSAGES
# ---------------------------------------------------------------------------
def check_if_goodbye_llm(llm, text: str) -> bool:
    """Utilise le LLM pour détecter si l'utilisateur souhaite clore la conversation."""
    if llm is None:
        return False

    if quota_is_exhausted():
        return False

    try:
        response = llm.invoke(CLOTURE_CLASSIFY_PROMPT.format(text=text)).content.strip().upper()
        return "OUI" in response
    except Exception as e:
        print(f"[CLÔTURE] ⚠️ Erreur LLM classification clôture : {e}")
        return False

def handle_trivial(text: str, llm=None, history: str = None) -> str | None:
    """Retourne une réponse si le message est purement trivial, sinon None."""
    # Étape 1a : Salutations directes
    if any(p.match(text) for p in GREETING_COMPILED):
        print(f"[FILTER] 👋 Salutation détectée : '{text[:50]}'")
        return GREETING_REPLY

    # Étape 1b : Pures banalités (merci, ok, émojis) -> Réponse automatique
    if any(p.match(text) for p in TRIVIAL_COMPILED):
        print(f"[FILTER] ⚠️ Trivial (regex) : '{text[:50]}'")
        if llm is None or quota_is_exhausted():
            return "Très bien ! Je reste disponible si vous avez des questions. 😊"
        try:
            response = llm.invoke(TRIVIAL_RESPONSE_PROMPT.format(text=text)).content.strip()
            print(f"[FILTER] 💬 Réponse (regex→LLM) : '{response[:80]}'")
            return response
        except Exception as e:
            print(f"[FILTER] ⚠️ Erreur LLM réponse triviale : {e}")
            return "Très bien ! Je reste disponible si vous avez des questions. 😊"

    # Sécurité contextuelle : By-pass si conversation active
    if history and "Aucune conversation précédente" not in history and len(history.strip()) > 0:
        print(f"[FILTER] 🧠 Session active détectée. Message envoyé directement au RAG.")
        return None

    # Étape 2 : Cas ambigus (uniquement pour le premier échange de la session)
    if llm is None or quota_is_exhausted():
        return None

    try:
        hist_str = history if history else "Aucun échange précédent."
        formatted_prompt = TRIVIAL_CLASSIFY_PROMPT.format(text=text, history=hist_str)
        response = llm.invoke(formatted_prompt).content.strip()
        
        if response.upper() == "QUESTION":
            print(f"[FILTER] ✅ Message pertinent (LLM) : '{text[:50]}'")
            return None
            
        print(f"[FILTER] ⚠️ Trivial (LLM) : '{text[:50]}' → '{response[:80]}'")
        return response
    except Exception as e:
        print(f"[FILTER] ⚠️ Erreur LLM classification : {e}")
        return None