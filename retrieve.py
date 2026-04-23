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

load_dotenv()


# Classe pour gérer les embeddings via l'API HuggingFace (modèle multilingue E5)
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


# Classe pour gérer la mémoire de conversation
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
        messages = self._get_user_memory(user_id)
        messages.append({
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


# Query expansion pour maximiser la couverture du retriever
def expand_query(llm, query: str) -> list[str]:
    """
    Demande au LLM de reformuler la question en variantes juridiques
    pour maximiser la couverture du retriever.
    """
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


# Récupération des documents pertinents avec filtrage par score de similarité et fallback intelligent
def retrieve_relevant_docs(vectorstore, queries: list[str], score_threshold: float = 1.2, max_docs: int = 12):
    """
    Cherche les documents pour chaque requête, filtre par score de similarité,
    et déduplique les résultats.
    """
    all_docs = []
    seen_contents = set()
    best_fallback = None
    best_fallback_score = float('inf')

    for q in queries:
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(q, k=8)

            for doc, score in docs_with_scores:
                # Garder le meilleur doc comme fallback au cas où tout est filtré
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
        print(f"[RETRIEVE] ⚠️ Aucun doc sous le seuil ({score_threshold}). Fallback sur le meilleur doc (score={best_fallback_score:.3f})")
        return [best_fallback], True  # True = fallback activé

    docs = [doc for doc, score in all_docs[:max_docs]]
    scores = [score for _, score in all_docs[:max_docs]]
    print(f"[RETRIEVE] ✅ {len(docs)} documents retenus — Scores: {[f'{s:.3f}' for s in scores]}")
    return docs, False


# Configuration du chatbot avec LLM, vectorstore, prompt et mémoire
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
        #gemma-3-27b-it
        temperature=0,
        google_api_key=gemini_key
    )

    template = """
        Tu es l'assistant expert de la HAAC (Haute Autorité de l'Audiovisuel et de la Communication) au Bénin.
        Ton rôle est de fournir des réponses précises et professionnelles basées UNIQUEMENT sur les documents officiels de la HAAC.

        INSTRUCTIONS STRICTES :
        1. Utilise UNIQUEMENT les informations du contexte fourni ci-dessous.
        2. N'ajoute JAMAIS d'informations issues de tes connaissances générales.
        3. Si le contexte ne mentionne PAS explicitement le sujet demandé, réponds UNIQUEMENT :
        "Je n'ai pas trouvé d'information spécifique sur ce point dans les documents officiels de la HAAC."
        N'extrais PAS d'informations adjacentes, générales ou vaguement liées pour compenser.
        4. N'inclus PAS de noms de fichiers dans ta réponse.
        5. Etoffe bien la réponse si le contexte le permet.
        6. Structure ta réponse ainsi :
        - Une phrase d'introduction courte
        - Des points numérotés pour les éléments distincts. Saute une ligne entre chaque point.
        - Des sous-points avec tirets (-) si nécessaire
        - Utilise *texte* pour mettre en valeur les termes importants

        CONTEXTE FOURNI :
        {context}

        HISTORIQUE ET QUESTION :
        {question}

        RÉPONSE :
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    memory = ConversationMemory(max_memory=4)

    return {
        "llm": llm,
        "vectorstore": vectorstore,
        "prompt": prompt,
        "memory": memory
    }


# Fonction principale pour poser une question avec mémoire et récupération de documents 
def poser_question_avec_memoire(chatbot_config, query, user_id=None):
    llm = chatbot_config["llm"]
    vectorstore = chatbot_config["vectorstore"]
    prompt = chatbot_config["prompt"]
    memory = chatbot_config["memory"]

    # 1. Historique
    history = memory.get_formatted_history(user_id)

    # 2. Expansion de la requête
    queries = expand_query(llm, query)

    # 3. Récupération avec filtre par score
    docs, is_fallback = retrieve_relevant_docs(vectorstore, queries)

    # 4. Construction du contexte
    if is_fallback:
        # Si on est en fallback, on signale au LLM que le contexte est peut-être hors sujet
        context = f"[Note : Les documents suivants sont les plus proches trouvés mais peuvent ne pas répondre directement à la question.]\n\n"
        context += "\n\n".join([
            f"Source: {d.metadata.get('source')}\nContenu: {d.page_content}" for d in docs
        ])
    else:
        context = "\n\n".join([
            f"Source: {d.metadata.get('source')}\nContenu: {d.page_content}" for d in docs
        ])

    # 5. Construction du prompt final
    input_data = {
        "context": context,
        "question": f"{history}\n\nQUESTION ACTUELLE: {query}"
    }

    # 6. Génération
    bot_response = llm.invoke(prompt.format(**input_data)).content

    # 7. Mise à jour mémoire
    memory.add_message("user", query, user_id)
    memory.add_message("assistant", bot_response, user_id)

    return {
        "response": bot_response,
        "sources": list(set([doc.metadata.get('source', 'Inconnue') for doc in docs]))
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