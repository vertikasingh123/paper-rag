"""Look at what retrieval actually returned for one question.

Aggregate metrics tell you something is broken. This tells you what.

Usage
-----
  python -m evals.inspect --pdf data/aiayn.pdf -q "How is positional information injected?"

  # replay a question from the gold set, with its evidence strings checked
  python -m evals.inspect --pdf data/aiayn.pdf --gold-id q07

  # also show what the LLM said, to separate retrieval bugs from answer bugs
  python -m evals.inspect --pdf data/aiayn.pdf --gold-id q10 --answer

  # search the raw chunks for a string, to check whether the text survived
  # extraction at all
  python -m evals.inspect --pdf data/aiayn.pdf --grep "sine and cosine"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evals.run_eval import normalise  # single definition, so it cannot drift
from rag.config import RAGConfig
from rag.pipeline import PaperRAG

ROOT = Path(__file__).resolve().parent


def show_chunk(text: str, width: int = 400) -> str:
    body = re.sub(r"\s+", " ", text).strip()
    return body if len(body) <= width else body[:width] + " …"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("-q", "--question")
    parser.add_argument("--gold", type=Path, default=ROOT / "gold_attention_v2.json")
    parser.add_argument("--gold-id", help="replay this question from the gold set")
    parser.add_argument("--grep", help="search the chunk text for a string instead")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--keep-references", action="store_true")
    parser.add_argument("--answer", action="store_true", help="also call the LLM")
    args = parser.parse_args()

    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        drop_references=not args.keep_references,
    )
    pipeline, cached = PaperRAG.from_pdf(args.pdf, config)
    print(f"{len(pipeline.chunks)} chunks{' (cached)' if cached else ''}\n")

    # -- grep mode: did the text survive extraction at all? ----------------
    if args.grep:
        needle = normalise(args.grep)
        found = False
        for chunk in pipeline.chunks:
            if needle in normalise(chunk.text):
                found = True
                print(f"chunk {chunk.id} · {chunk.page_label} · {chunk.section}")
                print(f"  {show_chunk(chunk.text, 600)}\n")
        if not found:
            print(f"NOT FOUND in any chunk: {args.grep!r}")
            print(
                "\nEither PDF extraction mangled it (check for ligatures, math "
                "notation or column interleaving), or your evidence string does "
                "not match the paper's wording. Either way this is a gold-set "
                "bug, not a retrieval failure."
            )
        return

    # -- resolve the question ---------------------------------------------
    evidence: list[str] = []
    expected: list[str] = []
    question = args.question

    if args.gold_id:
        gold = json.loads(args.gold.read_text(encoding="utf-8"))
        item = next(
            (q for q in gold["questions"] if q["id"] == args.gold_id), None
        )
        if item is None:
            raise SystemExit(f"No question with id {args.gold_id!r} in {args.gold}")
        question = item["question"]
        evidence = item.get("evidence", [])
        expected = item.get("answer_contains", [])
        if item.get("unanswerable"):
            print("(this question is marked unanswerable - it SHOULD abstain)\n")

    if not question:
        raise SystemExit("Pass either -q/--question or --gold-id.")

    print(f"Q: {question}\n")
    if evidence:
        print(f"looking for: {evidence}\n")

    # -- retrieval ---------------------------------------------------------
    sources = pipeline.retrieve(question)
    for source in sources:
        mark = ""
        if evidence:
            hit = any(normalise(e) in normalise(source.chunk.text) for e in evidence)
            mark = "  <-- CONTAINS EVIDENCE" if hit else ""
        print(f"[{source.rank}] score {source.score:.3f} · {source.label}{mark}")
        print(f"    {show_chunk(source.chunk.text)}\n")

    if evidence:
        ranks = [
            s.rank
            for s in sources
            if any(normalise(e) in normalise(s.chunk.text) for e in evidence)
        ]
        if ranks:
            print(f"evidence first appears at rank {ranks[0]}")
        else:
            print(
                "evidence NOT in the retrieved chunks.\n"
                "Next: run with --grep on one of the evidence strings. If it is "
                "not in ANY chunk, the problem is extraction or your gold set, "
                "not the retriever."
            )

    # -- answering ---------------------------------------------------------
    if args.answer:
        print("\n--- LLM answer ---")
        result = pipeline.answer(question)
        print(result.text)
        print(f"\nabstained: {result.abstained}")
        if expected:
            ok = any(normalise(e) in normalise(result.text) for e in expected)
            print(f"contains {expected}: {ok}")
            if not ok and sources:
                print(
                    "\nRetrieval worked but the answer missed. That is a prompt "
                    "or model problem, not a retrieval one - a different fix."
                )


if __name__ == "__main__":
    main()
