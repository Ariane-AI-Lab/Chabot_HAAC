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
def retrieve_relevant_docs(vectorstore, queries: list[str], score_threshold: float = 1.2, max_docs: int = 12):
    all_docs = []
    seen_contents = set()
    best_fallback = None
    best_fallback_score = float('inf')

    for q in queries:
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(q, k=8)
            for doc, score in docs_with_scores:
                if score < best_fallback_score:
                    best_fallback_score = score
                    best_fallback = doc
                if score <= score_threshold and doc.page_content not in seen_contents:
                    all_docs.append((doc, score))
                    seen_contents.add(doc.page_content)
        except Exception as e:
            print(f"[RETRIEVE] ⚠️ Erreur sur la requête '{q}' : {e}")

    all_docs.sort(key=lambda x: x[1])

    if not all_docs and best_fallback:
        print(f"[RETRIEVE] ⚠️ Aucun doc sous le seuil ({score_threshold}). Fallback (score={best_fallback_score:.3f})")
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
        model="gemma-3-27b-it",
        temperature=0,
        google_api_key=gemini_key
    )

    template = """
        Tu es l'assistant expert de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication) au Bénin.
        Ton rôle est de fournir des réponses précises et professionnelles basées sur les sources officielles de la HAAC.

        INSTRUCTIONS STRICTES :
        1. Tu disposes de DEUX sources de contexte complémentaires :
           - Les *documents officiels* (textes de loi, décrets, règlements) : fiables pour tout ce qui est juridique et procédural.
           - Le *site web haac.bj* (informations récentes) : fiable pour les personnes en poste, nominations, actualités.
        2. Stratégie de priorisation :
           - Pour les informations juridiques et réglementaires → privilégie les documents officiels.
           - Pour les personnes en poste, nominations, événements récents → privilégie le site web haac.bj.
           - Si une information est présente dans les deux sources, privilégie la plus récente (site web).
           - Si une source ne contient pas l'information, utilise l'autre sans hésiter.
        3. Si aucune des deux sources ne contient l'information demandée, réponds UNIQUEMENT :
           "Je n'ai pas trouvé d'information spécifique sur ce point dans les sources officielles de la HAAC."
        4. N'ajoute JAMAIS d'informations issues de tes connaissances générales.
        5. N'inclus PAS de noms de fichiers dans ta réponse.
        6. CONCISION — règle absolue :
           - Réponds UNIQUEMENT à ce qui est demandé, rien de plus.
           - Si on demande un NOM → donne uniquement le nom et le titre.
             Exemple : "Le président de la HAAC est *Edouard LOKO*."
           - Si on demande une LISTE → donne la liste, sans intro ni commentaire.
           - Si on demande une EXPLICATION ou une PROCÉDURE → structure avec des points numérotés,
             des sous-points avec tirets (-) si nécessaire, et *texte* pour les termes importants.
           - N'ajoute JAMAIS d'informations complémentaires (mandat, historique, autres fonctions, contexte)
             sauf si l'utilisateur le demande explicitement.
           - Supprime toute phrase de remplissage : "Selon les documents...", "Il est important de noter...", "En résumé..."

        CONTEXTE DOCUMENTS OFFICIELS (textes de loi, décrets, règlements) :
        {context_faiss}

        CONTEXTE SITE WEB HAAC (informations récentes) :
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

    # 1. Historique
    history = memory.get_formatted_history(user_id)

    # 2. Expansion de la requête
    queries = expand_query(llm, query)

    # 3. FAISS et Tavily en parallèle (les deux systématiquement)
    print(f"[FAISS] 🔍 Recherche dans les documents locaux...")
    t0 = time.time()
    docs = retrieve_relevant_docs(vectorstore, queries)
    print(f"[FAISS] ✅ Terminé en {time.time()-t0:.2f}s")

    print(f"[TAVILY] 🌐 Recherche sur haac.bj...")
    t1 = time.time()
    tavily_context = search_haac_website(query)
    print(f"[TAVILY] ✅ Terminé en {time.time()-t1:.2f}s — {len(tavily_context)} caractères récupérés")

    # 4. Construction des deux contextes séparés
    context_faiss = "\n\n".join([
        f"Source: {d.metadata.get('source')}\nContenu: {d.page_content}" for d in docs
    ]) if docs else "Aucun document pertinent trouvé dans les fichiers locaux."

    context_tavily = tavily_context if tavily_context else "Aucun résultat trouvé sur haac.bj."

    # 5. Construction du prompt final
    input_data = {
        "context_faiss": context_faiss,
        "context_tavily": context_tavily,
        "question": f"{history}\n\nQUESTION ACTUELLE: {query}"
    }

    # 6. Génération
    bot_response = llm.invoke(prompt.format(**input_data)).content

    # 7. Mise à jour mémoire
    memory.add_message("user", query, user_id)
    memory.add_message("assistant", bot_response, user_id)

    # 8. Sources combinées
    faiss_sources = list(set([doc.metadata.get('source', 'Inconnue') for doc in docs]))
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