import os
import os
import time
from huggingface_hub import InferenceClient
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = InferenceClient(provider="hf-inference", api_key=api_key)
        self.model = "intfloat/multilingual-e5-large"

    def _get_embedding(self, text):
        if not text.strip():
            return [0.0] * 1024  
        text = text.replace("\n", " ")
        for attempt in range(3):
            try:
                result = self.client.feature_extraction(text, model=self.model)
                # result est un numpy array, on le convertit en liste
                return result.tolist() if hasattr(result, 'tolist') else list(result)
            except Exception as e:
                if "429" in str(e):
                    print(f"⚠️ Rate limit. Pause 30s... (essai {attempt+1}/3)")
                    time.sleep(30)
                elif attempt == 2:
                    raise e
                else:
                    time.sleep(5)
        return [0.0] * 1024

    def embed_documents(self, texts):
        embeddings = []
        total = len(texts)
        for i, text in enumerate(texts):
            embeddings.append(self._get_embedding(text))
            if (i + 1) % 50 == 0:
                print(f"  ⏳ {i+1}/{total} chunks indexés...")
            time.sleep(0.3)
        return embeddings

    def embed_query(self, text):
        return self._get_embedding(text)


def preparer_documents(chemin_dossier):
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
    
    final_chunks = []
    
    if not os.path.exists(chemin_dossier):
        print(f"❌ Erreur: Le dossier {chemin_dossier} n'existe pas.")
        return []

    for file in os.listdir(chemin_dossier):
        if file.endswith(".md"):
            with open(os.path.join(chemin_dossier, file), "r", encoding="utf-8") as f:
                print(f"📖 Lecture de {file}...")
                content = f.read()
                sections = header_splitter.split_text(content)
                for doc in sections:
                    doc.metadata["source"] = file
                    sub_chunks = text_splitter.split_documents([doc])
                    final_chunks.extend(sub_chunks)
                    
    return final_chunks


# 1. Préparation
chunks = preparer_documents("markdowns")

# 2. Récupération de la clé Hugging Face (à ajouter dans votre .env)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("❌ Erreur: HF_TOKEN manquant dans le fichier .env")
else:
    # 3. Initialisation des nouveaux Embeddings
    embeddings = HuggingFaceAPIEmbeddings(api_key=hf_token)

    # 4. Création de la base
    print(f"📦 Indexation de {len(chunks)} morceaux dans FAISS via Hugging Face API...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Sauvegarde
    vectorstore.save_local("faiss_index_haac")
    print("✅ Base vectorielle créée avec succès !")