"""Configuration for the retrieval pipeline.

Everything that affects retrieval quality lives in one dataclass so the eval
harness can sweep over configs and produce comparable numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


@dataclass(frozen=True)
class RAGConfig:
    """Knobs that change what the retriever returns.

    Any field here is part of the cache key, so changing one invalidates the
    stored index automatically instead of silently serving stale chunks.
    """

    # --- chunking ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    # Splitters leave stragglers at section boundaries. A 25-character chunk
    # can still win a similarity search and then tells the model nothing, so
    # fold anything shorter than this into the chunk before it.
    min_chunk_chars: int = 120

    # --- text cleanup ---
    # Rejoin words broken across a line ("hyphen-\nation" -> "hyphenation").
    dehyphenate: bool = True
    # Drop the bibliography. It is ~20% of a typical paper and almost never
    # what a question is about, but it matches on author names and years.
    drop_references: bool = True

    # --- retrieval ---
    top_k: int = 5
    # Minimum cosine similarity for a chunk to be shown as a source.
    min_score: float = 0.0

    # --- models ---
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def fingerprint(self) -> str:
        """Short stable hash of the config, used in cache paths."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)
