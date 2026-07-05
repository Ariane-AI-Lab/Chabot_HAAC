import os
import time
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from collections import deque
from datetime import datetime
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# --- TAVILY : recherche web sur haac.bj (toujours appelé) ---
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_haac_website(query: str) -> str:
    """
    Recherche sur le site officiel haac.bj via Tavily.
    Appelé systématiquement pour apporter des infos fraîches en complément du FAISS.
    """
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            include_domains=["haac.bj"],
            max_results=2
        )
        context = ""
        for result in response.get('results', []):
            context += f"\nSource: {result['url']}\nContenu: {result['content']}\n"
        return context.strip()
    except Exception as e:
        print(f"[TAVILY] ⚠️ Erreur lors de la recherche : {e}")
        return ""


# --- CLASSE EMBEDDINGS ---
class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model = "intfloat/multilingual-e5-large"

    def _get_embedding(self, text):
        if not text.strip():
            return [0.0] * 1024
        text = text.replace("\n", " ")
        text_to_embed = f"query: {text}"

        for attempt in range(3):
            try:
                result = self.client.feature_extraction(text_to_embed, model=self.model)
                return result.tolist() if hasattr(result, 'tolist') else list(result)
            except Exception:
                time.sleep(2)
        return [0.0] * 1024

    def embed_documents(self, texts):
        return [self._get_embedding(t) for t in texts]

    def embed_query(self, text):
        return self._get_embedding(text)


# --- CLASSE MÉMOIRE (par utilisateur) ---
class ConversationMemory:
    def __init__(self, max_memory=4):
        self.max_memory = max_memory
        self.conversations = {}  # un deque par user_id

    def _get_user_memory(self, user_id):
        if user_id not in self.conversations:
            self.conversations[user_id] = deque(maxlen=self.max_memory)
        return self.conversations[user_id]

    def add_message(self, role, content, user_id=None):
        user_id = user_id or "default"
        self._get_user_memory(user_id).append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
        })

    def get_formatted_history(self, user_id=None):
        user_id = user_id or "default"
        messages = list(self._get_user_memory(user_id))
        if not messages:
            return "Aucune conversation précédente."
        formatted = "HISTORIQUE DE LA CONVERSATION :\n"
        for msg in messages:
            formatted += f"\n{msg['role'].upper()}: {msg['content']}"
        return formatted


def condense_query_with_history(llm, history_str: str, current_query: str) -> str:
    """
    Analyse l'historique et la question actuelle pour générer une requête de recherche 
    unique, complète et autonome pour le RAG.
    """
    if "Aucune conversation précédente" in history_str or not history_str.strip():
        return current_query

    condensation_prompt = f"""
    Tu es un ingénieur de recherche RAG. Ton rôle est de prendre un historique de conversation et une question actuelle pour en faire une REQUÊTE DE RECHERCHE autonome et ultra-précise.
    La requête finale doit contenir tous les mots-clés nécessaires (sujet, objet juridique, contexte béninois) pour chercher efficacement dans une base de données vectorielle.

    {history_str}

    QUESTION ACTUELLE: {current_query}

    Consigne : Génère uniquement la requête optimisée sous forme de mots-clés ou d'une phrase simple, sans introduction ni commentaire.
    Requête optimisée :"""
    
    try:
        response = llm.invoke(condensation_prompt).content.strip()
        print(f"[CONDENSE] 🧠 Requête contextualisée : '{response}'")
        return response
    except Exception as e:
        print(f"[CONDENSE] ⚠️ Erreur condensation : {e}")
        return current_query


# --- QUERY EXPANSION ---
def expand_query(llm, query: str) -> list[str]:
    expansion_prompt = f"""
        Tu es un expert juridique béninois spécialisé dans la réglementation des médias.
        Reformule la question suivante en 3 variantes courtes utilisant un vocabulaire juridique et officiel béninois (textes de loi, décrets, règlements, codes).
        Chaque variante doit aborder un angle différent de la question.

        Question originale: {query}

        Réponds UNIQUEMENT avec les 3 reformulations, une par ligne, sans numérotation ni tiret.
    """
    try:
        response = llm.invoke(expansion_prompt).content.strip()
        variants = [v.strip() for v in response.split('\n') if v.strip()]
        all_queries = [query] + variants[:3]
        print(f"[EXPAND] 🔄 Requêtes générées : {all_queries}")
        return all_queries
    except Exception as e:
        print(f"[EXPAND] ⚠️ Erreur expansion, utilisation de la requête originale : {e}")
        return [query]


# --- RÉCUPÉRATION FAISS AVEC SCORE ---
def retrieve_relevant_docs(vectorstore, queries, score_threshold=1.2, max_docs=8):
    all_docs = []
    seen_contents = set()
    best_fallback = None
    best_fallback_score = float('inf')

    def search_one(q):
        return vectorstore.similarity_search_with_score(q, k=4)

    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        results = list(executor.map(search_one, queries))

    for docs_with_scores in results:
        for doc, score in docs_with_scores:
            if score < best_fallback_score:
                best_fallback_score = score
                best_fallback = doc
            if score <= score_threshold and doc.page_content not in seen_contents:
                all_docs.append((doc, score))
                seen_contents.add(doc.page_content)

    all_docs.sort(key=lambda x: x[1])

    if not all_docs and best_fallback:
        print(f"[RETRIEVE] ⚠️ Fallback (score={best_fallback_score:.3f})")
        return [best_fallback]

    docs = [doc for doc, score in all_docs[:max_docs]]
    scores = [score for _, score in all_docs[:max_docs]]
    print(f"[RETRIEVE] ✅ {len(docs)} documents retenus — Scores: {[f'{s:.3f}' for s in scores]}")
    return docs


# --- CONFIGURATION DU CHATBOT ---
def configurer_chatbot():
    hf_token = os.getenv("HF_TOKEN")
    gemini_key = os.getenv("GENAI_API_KEY")

    embeddings = HuggingFaceAPIEmbeddings(api_key=hf_token)
    vectorstore = FAISS.load_local(
        "faiss_index_haac",
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=gemini_key
    )

    template = """
        Tu es l'assistant officiel de la HAAC (Haute Autorité de l'Audiovisuel 
        et de la Communication) au Bénin. Tu es chaleureux, professionnel et naturel 
        dans tes échanges, comme un agent de call center expérimenté.

        RÈGLES DE COMPORTEMENT :

        1. SALUTATIONS :
        - Regarde UNIQUEMENT "QUESTION ACTUELLE" pour décider de saluer ou non.
        - Si "QUESTION ACTUELLE" contient une salutation (bonjour, bonsoir, salut...) 
            → réponds à la salutation chaleureusement avant de répondre à la question.
        - Si "QUESTION ACTUELLE" ne contient PAS de salutation → ne salue JAMAIS, 
            réponds directement à la question.
        - Ne te base JAMAIS sur l'historique pour décider de saluer.

        2. GUIDAGE INTERACTIF (CHOIX MULTIPLES) :
        - Si la question de l'utilisateur est générale (ex: "Quelle est la procédure pour la carte de presse ?") et que les sources montrent que la procédure dépend de plusieurs situations distinctes (ex: Première délivrance, Renouvellement, Duplicata) :
          * Ne donne PAS toutes les listes d'un coup pour ne pas saturer l'écran.
          * Présente brièvement les options disponibles.
          * Demande explicitement et chaleureusement à l'utilisateur de préciser sa situation actuelle.
          * Exemple de ton : "La procédure pour obtenir la carte de presse dépend de votre situation. S'agit-il d'une *première demande*, d'un *renouvellement* ou d'une demande de *duplicata* ? Dites-moi ce qu'il en est pour que je vous donne la liste exacte des pièces ! 😊"

        3. QUESTIONS DE SUIVI :
        - Si l'utilisateur dit "cite le reste", "continue", "tu n'as pas tout dit", 
            "réponds à ma question" etc. → reprends le contexte de la conversation 
            précédente (HISTORIQUE) et complète ta réponse en donnant la suite des éléments.
        - Ne réponds JAMAIS "je n'ai pas trouvé d'information" à une relance 
            conversationnelle de ce type. C'est une faute grave.

        4. EXHAUSTIVITÉ STRICTE DES LISTES :
        - Si l'utilisateur demande une procédure, les pièces à fournir ou une liste (membres, articles, dossiers) OU s'il a répondu au choix de la Règle 2 (ex: "première demande") : Tu dois IMPÉRATIVEMENT lister TOUS les éléments présents dans les sources, du premier au dernier, sans aucune exception.
        - Il est formellement interdit de résumer, d'omettre des pièces ou de t'arrêter en milieu de liste. Chaque pièce manquante est considérée comme une fausse information pour l'usager.   
        - Si les sources contiennent des informations partiellement liées à la question, exploite-les au maximum plutôt que de dire que tu n'as pas trouvé.
        - Interdiction absolue de répondre "je n'ai pas trouvé" quand l'information est présente dans les sources, même partiellement. 

        5. UTILISATION DES SOURCES :
        - Tu disposes de DEUX sources complémentaires :
            * Documents officiels (lois, décrets, règlements) → pour tout ce qui est juridique et procédural.
            * Site web haac.bj → pour les personnes en poste, nominations, actualités.
        - Utilise toutes les informations disponibles dans les deux sources.

        6. PROBLÈMES TECHNIQUES :
        - Si l'utilisateur signale un problème technique (plateforme en panne, bug, accès impossible...) réponds précisément :
            "Je ne suis pas spécialisé dans la résolution des problèmes techniques. Pour toute assistance technique, veuillez contacter la HAAC directement :
            📧 contact@haac.bj
            📞 +229 XX XX XX XX
            Nos équipes se feront un plaisir de vous aider."

        7. ABSENCE D'INFORMATION COMPLÈTE OU CONFUSION :
        - Si la question porte sur un sujet institutionnel ou réglementaire de la HAAC, mais qu'après vérification rigoureuse du CONTEXTE DOCUMENTS OFFICIELS et du CONTEXTE SITE WEB HAAC, tu ne trouves ABSOLUMENT AUCUNE information concrète ou partielle pour y répondre, applique immédiatement la Règle 9 ci-dessous.

        8. TON ET FORMATAGE (STYLE CALL CENTER) :
        - Reste naturel, courtois et humain — évite absolument les formules robotiques.
        - Utilise les astérisques (*texte*) pour mettre en valeur les termes importants.
        - Structure tes réponses avec des points numérotés clairs ou des puces (•) pour faciliter la lecture sur WhatsApp.
        - Ne sacrifie JAMAIS l'exactitude ou l'exhaustivité juridique pour faire court. Si la liste officielle est longue, donne-la entièrement.

        9. RÈGLE CRITIQUE : BASCULE ET PASSATION HUMAINE :
        - Si et seulement si les sources fournies (FAISS et Tavily) sont muettes, insuffisantes, ou contradictoires sur la demande de l'utilisateur, tu dois impérativement générer le signal exact suivant : [TRIGGER_HANDOVER]
        - Ne rajoute aucun commentaire, aucune phrase d'excuse ou de politesse autour. Écris UNIQUEMENT ce code secret. Le système backend se chargera de le capter pour le transférer à un humain.

        CONTEXTE DOCUMENTS OFFICIELS :
        {context_faiss}

        CONTEXTE SITE WEB HAAC :
        {context_tavily}

        HISTORIQUE ET QUESTION :
        {question}

        RÉPONSE :
"""
    prompt = PromptTemplate(template=template, input_variables=["context_faiss", "context_tavily", "question"])
    memory = ConversationMemory(max_memory=4)

    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "prompt": prompt,
        "memory": memory
    }


# --- FONCTION PRINCIPALE ---
def poser_question_avec_memoire(chatbot_config, query, user_id=None):
    llm = chatbot_config["llm"]
    vectorstore = chatbot_config["vectorstore"]
    prompt = chatbot_config["prompt"]
    memory = chatbot_config["memory"]

    history = memory.get_formatted_history(user_id)

    # 1b. Condensation contextuelle de la requête
    print(f"[PIPELINE] 🧠 Analyse du contexte conversationnel...")
    search_query = condense_query_with_history(llm, history, query)

    # 2. Expansion + Tavily en parallèle
    print(f"[PIPELINE] 🚀 Lancement parallèle : expansion + Tavily...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_queries = executor.submit(expand_query, llm, search_query)
        future_tavily  = executor.submit(search_haac_website, search_query)

        queries        = future_queries.result()
        tavily_context = future_tavily.result()

    print(f"[PIPELINE] ✅ Expansion + Tavily terminés en {time.time()-t0:.2f}s")
    print(f"[TAVILY] 📄 {len(tavily_context)} caractères récupérés")

    # 3. FAISS
    print(f"[FAISS] 🔍 Recherche dans les documents locaux...")
    t1 = time.time()
    docs = retrieve_relevant_docs(vectorstore, queries)
    print(f"[FAISS] ✅ Terminé en {time.time()-t1:.2f}s")

    # 4. Construction des deux contextes séparés
    context_faiss = "\n\n".join([
        f"Source: {d.metadata.get('source')}\nContenu: {d.page_content}" for d in docs
    ]) if docs else ""

    context_tavily = tavily_context if tavily_context else ""

    # 5. Construction du prompt final
    input_data = {
        "context_faiss": context_faiss,
        "context_tavily": context_tavily,
        "question": f"{history}\n\nQUESTION ACTUELLE: {query}"
    }

    # 6. Génération
    bot_response = llm.invoke(prompt.format(**input_data)).content.strip()

    # 7. Mise à jour mémoire (Uniquement si ce n'est pas un trigger de handover)
    if "[TRIGGER_HANDOVER]" not in bot_response:
        memory.add_message("user", query, user_id)
        memory.add_message("assistant", bot_response, user_id)

    # 8. Sources combinées
    faiss_sources = list(set([doc.metadata.get('source', 'Inconnue') for doc in docs])) if docs else []
    tavily_sources = ["haac.bj (web)"] if tavily_context else []

    return {
        "response": bot_response,
        "sources": faiss_sources + tavily_sources
    }


# --- MODE CLI ---
if __name__ == "__main__":
    mon_bot = configurer_chatbot()
    print("--- Chatbot HAAC (Tapez 'quit' pour sortir) ---")
    while True:
        user_input = input("\nVotre question : ")
        if user_input.lower() == 'quit':
            break

        result = poser_question_avec_memoire(mon_bot, user_input)
        print(f"\n🤖 RÉPONSE :\n{result['response']}")
        print(f"\n📚 SOURCES : {', '.join(result['sources'])}")