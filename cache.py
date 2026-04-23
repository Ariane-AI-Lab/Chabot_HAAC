import os
import json
import time
import numpy as np
import faiss
from datetime import datetime

# ---------------------------------------------------------------------------
# CACHE SÉMANTIQUE GLOBAL (partagé entre tous les utilisateurs)
# ---------------------------------------------------------------------------
# Fonctionnement :
#   1. À chaque nouvelle question, on calcule son embedding
#   2. On cherche dans le FAISS dédié si une question similaire existe
#   3. Si score de similarité > SIMILARITY_THRESHOLD → on retourne la réponse cachée
#   4. Sinon → pipeline normal, puis on ajoute la paire (question, réponse) au cache
#
# Stockage :
#   - faiss_cache/cache.index   : les vecteurs des questions
#   - faiss_cache/cache.json    : les métadonnées (question, réponse, sources, date)
# ---------------------------------------------------------------------------

CACHE_DIR = "faiss_cache"
CACHE_INDEX_PATH = os.path.join(CACHE_DIR, "cache.index")
CACHE_META_PATH = os.path.join(CACHE_DIR, "cache.json")

EMBEDDING_DIM = 1024          # Doit correspondre au modèle E5
SIMILARITY_THRESHOLD = 0.92   # Entre 0 et 1 — plus élevé = plus strict
                               # 0.92 : questions quasi-identiques seulement
                               # 0.85 : questions similaires acceptées
MAX_CACHE_SIZE = 500           # Nombre max d'entrées (FIFO si dépassé)


class SemanticCache:
    def __init__(self, embeddings_fn):
        """
        embeddings_fn : callable qui prend un str et retourne list[float]
                        (typiquement HuggingFaceAPIEmbeddings.embed_query)
        """
        self.embeddings_fn = embeddings_fn
        self.index = None
        self.metadata: list[dict] = []
        self._load_or_create()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _load_or_create(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        if os.path.exists(CACHE_INDEX_PATH) and os.path.exists(CACHE_META_PATH):
            print("[CACHE] 📂 Chargement du cache existant...")
            self.index = faiss.read_index(CACHE_INDEX_PATH)
            with open(CACHE_META_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            print(f"[CACHE] ✅ {len(self.metadata)} entrées chargées")
        else:
            print("[CACHE] 🆕 Création d'un nouveau cache...")
            # IndexFlatIP = produit scalaire (cosine similarity si vecteurs normalisés)
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.metadata = []
            self._save()

    def _save(self):
        faiss.write_index(self.index, CACHE_INDEX_PATH)
        with open(CACHE_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # ── Normalisation L2 (nécessaire pour cosine similarity avec IndexFlatIP) ─

    @staticmethod
    def _normalize(vector: list[float]) -> np.ndarray:
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm

    # ── Recherche ────────────────────────────────────────────────────────────

    def get(self, question: str) -> dict | None:
        """
        Cherche une réponse en cache pour la question donnée.
        Retourne un dict {response, sources, cached_question} si trouvé, sinon None.
        """
        if self.index.ntotal == 0:
            return None

        t0 = time.time()
        embedding = self.embeddings_fn(question)
        vector = self._normalize(embedding).reshape(1, -1)

        scores, indices = self.index.search(vector, k=1)
        score = float(scores[0][0])
        idx = int(indices[0][0])

        print(f"[CACHE] 🔍 Meilleur score de similarité : {score:.4f} (seuil={SIMILARITY_THRESHOLD})")

        if score >= SIMILARITY_THRESHOLD and 0 <= idx < len(self.metadata):
            meta = self.metadata[idx]
            print(f"[CACHE] ✅ HIT en {time.time()-t0:.2f}s — Question originale : '{meta['question'][:80]}'")
            return {
                "response": meta["response"],
                "sources": meta["sources"],
                "cached_question": meta["question"]
            }

        print(f"[CACHE] ❌ MISS en {time.time()-t0:.2f}s")
        return None

    # ── Ajout ────────────────────────────────────────────────────────────────

    def add(self, question: str, response: str, sources: list[str]):
        """
        Ajoute une nouvelle paire (question → réponse) au cache.
        Applique un FIFO si le cache dépasse MAX_CACHE_SIZE.
        """
        # Éviter les doublons exacts
        existing = self.get(question)
        if existing and existing["cached_question"].strip().lower() == question.strip().lower():
            print("[CACHE] ⚠️ Question déjà en cache, pas d'ajout")
            return

        embedding = self.embeddings_fn(question)
        vector = self._normalize(embedding).reshape(1, -1)

        # FIFO : supprimer la plus ancienne entrée si cache plein
        if self.index.ntotal >= MAX_CACHE_SIZE:
            print(f"[CACHE] 🗑️ Cache plein ({MAX_CACHE_SIZE}), suppression de la plus ancienne entrée")
            self._remove_oldest()

        self.index.add(vector)
        self.metadata.append({
            "question": question,
            "response": response,
            "sources": sources,
            "added_at": datetime.now().isoformat()
        })
        self._save()
        print(f"[CACHE] ➕ Entrée ajoutée ({self.index.ntotal} total)")

    def _remove_oldest(self):
        """
        FAISS IndexFlatIP ne supporte pas la suppression directe.
        On recrée l'index sans la première entrée (la plus ancienne).
        """
        if len(self.metadata) == 0:
            return

        self.metadata.pop(0)

        # Reconstruire l'index sans le premier vecteur
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if len(self.metadata) > 0:
            embeddings = [self.embeddings_fn(m["question"]) for m in self.metadata]
            vectors = np.array([self._normalize(e) for e in embeddings], dtype=np.float32)
            new_index.add(vectors)

        self.index = new_index

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_entries": self.index.ntotal,
            "max_size": MAX_CACHE_SIZE,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cache_dir": CACHE_DIR
        }