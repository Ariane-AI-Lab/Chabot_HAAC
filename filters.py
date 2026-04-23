import re
import time
from collections import defaultdict

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
# CONSTANTES TRIVIAUX
# ---------------------------------------------------------------------------

# Salutations : réponse fixe hardcodée, pas besoin du LLM
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

# Autres cas triviaux évidents : le LLM génère une réponse naturelle adaptée
TRIVIAL_PATTERNS = [
    r"^\s*(merci|thanks|thx|thank you)\s*[!?.]*\s*$",
    r"^\s*(ok|okay|👍|👎|😊|🙏)\s*$",
    r"^\s*.{0,2}\s*$",
    r"^\s*(test|testing|essai|123|ping)\s*[!?.]*\s*$",
]

TRIVIAL_COMPILED = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in TRIVIAL_PATTERNS]

# Prompt utilisé quand la regex a DÉJÀ confirmé que c'est trivial
# → on ne reclassifie pas, on génère juste une réponse naturelle
TRIVIAL_RESPONSE_PROMPT = """Tu es l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication du Bénin).

Un utilisateur t'envoie ce message : "{text}"

Génère une réponse courte (1 à 2 phrases maximum), naturelle, chaleureuse et adaptée au contexte exact du message.
Ne te présente pas sauf si c'est une salutation d'ouverture. Réponds juste de manière humaine et cohérente.

Exemples :
- "Bonjour" → "👋 Bonjour ! Je suis l'assistant officiel de la HAAC. En quoi puis-je vous aider ?"
- "Merci" → "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."
- "Ok" → "Très bien ! Je reste disponible si vous avez des questions. 😊"
- "Au revoir" → "À bientôt ! N'hésitez pas à revenir si vous avez besoin d'informations. 👋"

Réponds UNIQUEMENT avec la réponse courte, rien d'autre."""

# Prompt utilisé pour les cas AMBIGUS non détectés par regex
# → le LLM classifie ET génère la réponse si trivial
TRIVIAL_CLASSIFY_PROMPT = """Tu es l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication du Bénin).

Un utilisateur t'envoie ce message : "{text}"

Ce message contient-il une vraie question ou demande d'information sur la HAAC, la réglementation audiovisuelle, les médias ou les procédures officielles ?
- Si OUI → réponds UNIQUEMENT avec le mot : QUESTION
- Si NON → génère une réponse courte (1 à 2 phrases max), naturelle et adaptée au contexte. Ne te présente pas.

Exemples de messages NON pertinents et leurs réponses :
- "Ok c'est compris" → "Très bien ! Je reste disponible si vous avez des questions. 😊"
- "Je vais vous revenir" → "Pas de souci, je serai là ! 👋"
- "je t'ai pas encore posé une question" → "Pas de problème, prenez votre temps ! Je suis là quand vous êtes prêt. 😊"

Réponds UNIQUEMENT avec QUESTION ou avec la réponse courte, rien d'autre."""


# ---------------------------------------------------------------------------
# ANTI-SPAM
# ---------------------------------------------------------------------------

user_message_times: dict[str, list[float]] = defaultdict(list)


def is_rate_limited(sender_id: str) -> bool:
    """
    Vérifie si l'utilisateur dépasse le seuil de messages autorisés.

    Returns:
        True si l'utilisateur est bloqué, False sinon.
    """
    now = time.time()
    times = [t for t in user_message_times[sender_id] if now - t < SPAM_WINDOW_SECONDS]
    user_message_times[sender_id] = times

    if len(times) >= SPAM_MAX_MESSAGES:
        print(f"[SPAM] 🚫 {sender_id} bloqué ({len(times)} msgs en {SPAM_WINDOW_SECONDS}s)")
        return True

    user_message_times[sender_id].append(now)
    return False


# ---------------------------------------------------------------------------
# DÉTECTION ET RÉPONSE CONTEXTUELLE AUX MESSAGES TRIVIAUX
# ---------------------------------------------------------------------------

def handle_trivial(text: str, llm=None) -> str | None:
    """
    Retourne une réponse contextuelle et naturelle si le message est trivial,
    sinon retourne None pour laisser passer au pipeline RAG.

    Stratégie en deux étapes distinctes :
    1. Regex détecte les cas évidents → LLM génère la réponse SANS reclassifier
    2. Cas ambigus → LLM classifie ET génère la réponse si trivial

    Args:
        text (str): Le message de l'utilisateur.
        llm: L'instance LLM du chatbot (Gemini).

    Returns:
        str | None: Une réponse courte et naturelle si trivial, None sinon.
    """

    # Étape 1a : salutations — réponse fixe, aucun appel LLM
    if any(p.match(text) for p in GREETING_COMPILED):
        print(f"[FILTER] 👋 Salutation détectée : '{text[:50]}'")
        return GREETING_REPLY

    # Étape 1b : autres triviaux évidents — LLM génère une réponse naturelle (sans classifier)
    if any(p.match(text) for p in TRIVIAL_COMPILED):
        print(f"[FILTER] ⚠️ Trivial (regex) : '{text[:50]}'")
        if llm is None:
            return "Très bien ! Je reste disponible si vous avez des questions. 😊"
        try:
            response = llm.invoke(TRIVIAL_RESPONSE_PROMPT.format(text=text)).content.strip()
            print(f"[FILTER] 💬 Réponse (regex→LLM) : '{response[:80]}'")
            return response
        except Exception as e:
            print(f"[FILTER] ⚠️ Erreur LLM réponse triviale : {e}")
            return "Très bien ! Je reste disponible si vous avez des questions. 😊"

    # Étape 2 : cas ambigus — LLM classifie ET génère si trivial
    if llm is None:
        return None

    try:
        response = llm.invoke(TRIVIAL_CLASSIFY_PROMPT.format(text=text)).content.strip()
        if response.upper() == "QUESTION":
            print(f"[FILTER] ✅ Message pertinent (LLM) : '{text[:50]}'")
            return None  # → pipeline RAG normal
        print(f"[FILTER] ⚠️ Trivial (LLM) : '{text[:50]}' → '{response[:80]}'")
        return response
    except Exception as e:
        print(f"[FILTER] ⚠️ Erreur LLM classification : {e}")
        return None  # En cas d'erreur, on laisse passer au RAG