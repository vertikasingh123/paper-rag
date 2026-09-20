"""Local embeddings + a FAISS index, with an on-disk cache.

Embeddings run on your machine via sentence-transformers, so indexing a paper
costs nothing and works offline. Only the answer step calls an API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

from .config import RAGConfig
from .ingest import Chunk

CACHE_DIR = Path(".cache")

_MODEL_CACHE: dict[str, object] = {}


def get_embedder(model_name: str):
    """Load the embedding model once per process (it is ~90MB and slow to init)."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """Embed and L2-normalise, so an inner product IS cosine similarity."""
    model = get_embedder(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


class VectorStore:
    """A FAISS index plus the chunks it points at."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk], config: RAGConfig):
        self.index = index
        self.chunks = chunks
        self.config = config

    # -- build / persist ---------------------------------------------------

    @classmethod
    def build(cls, chunks: list[Chunk], config: RAGConfig) -> "VectorStore":
        if not chunks:
            raise ValueError("No chunks to index - the PDF produced no usable text.")
        vectors = embed_texts([c.text for c in chunks], config.embedding_model)
        # IndexFlatIP over normalised vectors = exact cosine search. At paper
        # scale (a few hundred chunks) exact beats approximate on every axis.
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks, config)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        payload = {
            "config": self.config.to_dict(),
            "chunks": [c.to_dict() for c in self.chunks],
        }
        (directory / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, config: RAGConfig) -> "VectorStore":
        index = faiss.read_index(str(directory / "index.faiss"))
        payload = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(d) for d in payload["chunks"]]
        return cls(index, chunks, config)

    # -- search ------------------------------------------------------------

    def search(self, question: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        k = top_k or self.config.top_k
        k = min(k, len(self.chunks))
        query = embed_texts([question], self.config.embedding_model)
        scores, ids = self.index.search(query, k)
        results: list[tuple[Chunk, float]] = []
        for chunk_id, score in zip(ids[0], scores[0]):
            if chunk_id == -1:
                continue
            if score < self.config.min_score:
                continue
            results.append((self.chunks[int(chunk_id)], float(score)))
        return results


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


def file_fingerprint(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def cache_path(pdf_fingerprint: str, config: RAGConfig, root: Path = CACHE_DIR) -> Path:
    """Cache key covers the file AND the config, so a knob change rebuilds."""
    return root / f"{pdf_fingerprint}-{config.fingerprint()}"


def load_or_build(
    chunks_factory, pdf_fingerprint: str, config: RAGConfig, root: Path = CACHE_DIR
) -> tuple["VectorStore", bool]:
    """Return (store, was_cached). `chunks_factory` is only called on a miss."""
    directory = cache_path(pdf_fingerprint, config, root)
    if (directory / "index.faiss").exists():
        try:
            return VectorStore.load(directory, config), True
        except Exception:
            pass  # corrupt cache entry - fall through and rebuild
    store = VectorStore.build(chunks_factory(), config)
    store.save(directory)
    return store, False
