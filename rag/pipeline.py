"""Retrieve, then answer with citations - and abstain when the paper is silent.

The abstention rule is deliberate. A paper assistant that confidently answers
from its own pretraining, when the retrieved context says nothing, is worse
than useless: the user cannot tell which sentences came from their PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RAGConfig
from .ingest import Chunk, load_pdf
from .llm import LLMClient
from .store import VectorStore, cache_path, file_fingerprint, load_or_build

SYSTEM_PROMPT = """You are a research assistant answering questions about ONE paper.

Rules:
1. Use only the numbered excerpts provided. Do not use outside knowledge, even
   if you recognise the paper.
2. Cite the excerpt number in square brackets after each claim, like [2].
   A sentence that makes a claim with no bracket is a bug.
3. If the excerpts do not contain the answer, say exactly:
   "The excerpts provided don't cover this." Then, in one sentence, say what
   section of the paper would likely have it. Do not guess the answer.
4. Quote the paper's own numbers and terms rather than paraphrasing them.
5. Be concise. Two or three sentences unless the question asks for more."""

ANSWER_TEMPLATE = """Excerpts from the paper:

{context}

---
Question: {question}

Answer using only the excerpts above, citing them as [n]."""


@dataclass
class Source:
    rank: int
    chunk: Chunk
    score: float

    @property
    def label(self) -> str:
        bits = [self.chunk.page_label]
        if self.chunk.section:
            bits.append(self.chunk.section)
        return " · ".join(bits)


@dataclass
class Answer:
    text: str
    sources: list[Source]
    question: str

    @property
    def abstained(self) -> bool:
        return "don't cover this" in self.text.lower()


class PaperRAG:
    """The whole pipeline for a single paper."""

    def __init__(self, store: VectorStore, config: RAGConfig, llm: LLMClient | None = None):
        self.store = store
        self.config = config
        self._llm = llm

    # -- construction ------------------------------------------------------

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str | Path,
        config: RAGConfig | None = None,
        llm: LLMClient | None = None,
        use_cache: bool = True,
    ) -> tuple["PaperRAG", bool]:
        """Build (or load from cache) a pipeline for one PDF.

        Returns (pipeline, was_cached).
        """
        config = config or RAGConfig()
        pdf_path = Path(pdf_path)
        fingerprint = file_fingerprint(pdf_path.read_bytes())

        def factory() -> list[Chunk]:
            return load_pdf(pdf_path, config)

        if use_cache:
            store, cached = load_or_build(factory, fingerprint, config)
        else:
            store, cached = VectorStore.build(factory(), config), False
        return cls(store, config, llm), cached

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @property
    def chunks(self) -> list[Chunk]:
        return self.store.chunks

    # -- query -------------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[Source]:
        hits = self.store.search(question, top_k)
        return [Source(rank=i + 1, chunk=c, score=s) for i, (c, s) in enumerate(hits)]

    def answer(
        self, question: str, top_k: int | None = None, temperature: float = 0.1
    ) -> Answer:
        """Pass temperature=0.0 from the eval harness. Sampling noise makes
        answer accuracy vary run to run, which is not what you want to be
        measuring when comparing two retrieval configs."""
        sources = self.retrieve(question, top_k)
        if not sources:
            return Answer(
                text="The excerpts provided don't cover this. Nothing in the "
                "indexed text matched the question.",
                sources=[],
                question=question,
            )
        context = "\n\n".join(
            f"[{s.rank}] ({s.label})\n{s.chunk.text}" for s in sources
        )
        prompt = ANSWER_TEMPLATE.format(context=context, question=question)
        text = self.llm.complete(SYSTEM_PROMPT, prompt, temperature=temperature)
        return Answer(text=text, sources=sources, question=question)

    # -- housekeeping ------------------------------------------------------

    def cache_location(self, pdf_fingerprint: str) -> Path:
        return cache_path(pdf_fingerprint, self.config)
