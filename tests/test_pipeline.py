"""Offline smoke test. No API key, no model download, no network.

Run with:  python tests/test_pipeline.py

The embedding model is replaced by a deterministic hashing vectoriser, so
this exercises chunking, page provenance, FAISS, the cache and the retrieval
plumbing without waiting on a 90MB download. Retrieval *quality* is not what
this measures - that is what evals/run_eval.py is for.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import store as store_module  # noqa: E402
from rag.config import RAGConfig  # noqa: E402
from rag.ingest import clean_text, load_pdf  # noqa: E402
from rag.pipeline import PaperRAG  # noqa: E402
from tests.make_sample_pdf import build  # noqa: E402

DIM = 256


class HashingEmbedder:
    """Bag-of-words hashed into a fixed vector. Crude, but deterministic and
    good enough that a keyword query retrieves the chunk containing it."""

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), DIM), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9.]+", text.lower()):
                # crc32, not hash(), so results do not depend on PYTHONHASHSEED
                vectors[row, zlib.crc32(token.encode()) % DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-9, None)


def install_stub() -> None:
    store_module._MODEL_CACHE["stub"] = HashingEmbedder()


PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" - {detail}" if detail and not condition else ""))


def main() -> int:
    install_stub()
    pdf = build()
    config = RAGConfig(
        chunk_size=700, chunk_overlap=120, top_k=3, embedding_model="stub"
    )

    print("\ningest")
    raw = "The popu-\nlation prior is es-\ntimated using a non-\nGaussian model."
    cleaned = clean_text(raw)
    check("rejoins hyphenated line breaks", "population" in cleaned, cleaned)
    check("keeps hyphens before capitals", "non-Gaussian" in cleaned, cleaned)

    chunks = load_pdf(pdf, config)
    check("produces chunks", len(chunks) > 0, f"got {len(chunks)}")
    check(
        "every chunk knows its page",
        all(chunk.pages for chunk in chunks),
    )
    check(
        "pages are within the document",
        all(max(c.pages) <= 2 for c in chunks),
        str(sorted({p for c in chunks for p in c.pages})),
    )
    check(
        "detects section headings",
        any(c.section and "Experiments" in c.section for c in chunks),
        str({c.section for c in chunks}),
    )
    check(
        "drops the bibliography",
        not any("Almeida" in c.text for c in chunks),
    )
    check(
        "no straggler chunks",
        all(len(c.text) >= config.min_chunk_chars for c in chunks),
        str(sorted(len(c.text) for c in chunks)[:3]),
    )
    check(
        "keeps the body text",
        any("SENTINEL" in c.text for c in chunks)
        and any("0.87" in c.text for c in chunks),
    )

    print("\nindex and retrieval")
    workdir = Path(tempfile.mkdtemp())
    try:
        pipeline, cached = PaperRAG.from_pdf(pdf, config, use_cache=False)
        check("builds an index", len(pipeline.chunks) == len(chunks))
        check("reports a cold build", cached is False)

        hits = pipeline.retrieve("SENTINEL cohort patients readings")
        check("returns results", len(hits) > 0, f"got {len(hits)}")
        check(
            "finds the data section",
            any("SENTINEL" in h.chunk.text for h in hits),
            str([h.label for h in hits]),
        )
        check(
            "scores are descending",
            all(
                hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)
            ),
        )
        check("respects top_k", len(hits) <= config.top_k)
        check(
            "labels sources with a page",
            all(h.label.startswith("p") for h in hits),
            str([h.label for h in hits]),
        )

        print("\ncache")
        store_module.CACHE_DIR = workdir
        first, cold = PaperRAG.from_pdf(pdf, config)
        second, warm = PaperRAG.from_pdf(pdf, config)
        check("first build is cold", cold is False)
        check("second build is served from cache", warm is True)
        check(
            "cached index returns the same chunks",
            [c.text for c in first.chunks] == [c.text for c in second.chunks],
        )

        other = RAGConfig(
            chunk_size=400, chunk_overlap=0, top_k=3, embedding_model="stub"
        )
        _, changed = PaperRAG.from_pdf(pdf, other)
        check("a config change invalidates the cache", changed is False)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        store_module.CACHE_DIR = Path(".cache")

    print("\neval scorer")
    from evals.run_eval import contains_any, score_retrieval

    check(
        "evidence matching ignores case and whitespace",
        contains_any("an F1 of   0.87 on the\nheld-out set", ["f1 of 0.87"]),
    )
    gold = {
        "questions": [
            {
                "id": "t1",
                "question": "How many patients are in the cohort?",
                "evidence": ["1,284 patients"],
                "answer_contains": ["1,284"],
            },
            {"id": "t2", "question": "What is the price of tea?", "unanswerable": True},
        ]
    }
    result = score_retrieval(pipeline, gold, top_k=3)
    check("scores only answerable questions", result.n == 1)
    check("metrics are in range", 0.0 <= result.hit_at_k <= 1.0 and 0.0 <= result.mrr <= 1.0)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
