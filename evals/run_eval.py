"""Measure the pipeline instead of eyeballing it.

Two things get scored, and they fail for different reasons:

  RETRIEVAL   Did the right text make it into the context window?
              Metrics: hit@k, MRR. No API key needed, runs in seconds.
              If this is low, no prompt will save you.

  ANSWERING   Given the right text, did the model say the right thing, and
              did it stay quiet when the paper is silent?
              Metrics: answer accuracy, false-answer rate on the
              deliberately unanswerable questions. Needs an API key.

Usage
-----
  # retrieval only, sweep the chunking config to find a better one
  python -m evals.run_eval --pdf data/attention.pdf --sweep

  # full run with the current config, including the LLM answer step
  python -m evals.run_eval --pdf data/attention.pdf --answers
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import RAGConfig
from rag.pipeline import PaperRAG

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


# Characters a model or a PDF substitutes for a plain hyphen or space.
# Missing these silently turns a correct answer into a failed test: gpt-oss
# returns U+2011 NON-BREAKING HYPHEN inside "byte-pair", which NFKC does not
# fold, so it has to be handled explicitly.
_DASHES = dict.fromkeys(
    map(ord, "‐‑‒–—―−­"), "-"
)
_SPACES = dict.fromkeys(
    map(ord, "       "), " "
)


def normalise(text: str) -> str:
    """Lowercase, ASCII punctuation, collapsed whitespace, no markdown."""
    text = unicodedata.normalize("NFKC", text.lower())
    text = text.translate(_DASHES).translate(_SPACES)
    text = text.replace("*", "")
    return re.sub(r"\s+", " ", text)


def contains_any(haystack: str, needles: list[str]) -> bool:
    hay = normalise(haystack)
    return any(normalise(n) in hay for n in needles)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    hit_at_k: float
    mrr: float
    misses: list[str]
    n: int
    # tier -> {"hit_at_k", "mrr", "n"}. A blended score hides the fact that
    # lookups are trivially easy and multi-hop questions are not.
    by_tier: dict = field(default_factory=dict)


def score_retrieval(pipeline: PaperRAG, gold: dict, top_k: int) -> RetrievalResult:
    answerable = [q for q in gold["questions"] if not q.get("unanswerable")]
    hits = 0
    reciprocal = 0.0
    misses: list[str] = []
    tiers: dict[str, list[float]] = defaultdict(list)

    for item in answerable:
        sources = pipeline.retrieve(item["question"], top_k=top_k)
        rank = None
        for source in sources:
            if contains_any(source.chunk.text, item["evidence"]):
                rank = source.rank
                break
        if rank:
            hits += 1
            reciprocal += 1.0 / rank
        else:
            misses.append(item["id"])
        tiers[item.get("tier", "all")].append(1.0 / rank if rank else 0.0)

    n = len(answerable) or 1
    by_tier = {
        tier: {
            "n": len(scores),
            "hit_at_k": sum(1 for s in scores if s > 0) / len(scores),
            "mrr": sum(scores) / len(scores),
        }
        for tier, scores in sorted(tiers.items())
    }
    return RetrievalResult(hits / n, reciprocal / n, misses, len(answerable), by_tier)


@dataclass
class AnswerResult:
    accuracy: float
    false_answer_rate: float
    wrong: list[str]
    hallucinated: list[str]


def score_answers(pipeline: PaperRAG, gold: dict, sleep: float = 1.0) -> AnswerResult:
    answerable = [q for q in gold["questions"] if not q.get("unanswerable")]
    unanswerable = [q for q in gold["questions"] if q.get("unanswerable")]

    correct, wrong = 0, []
    for item in answerable:
        result = pipeline.answer(item["question"], temperature=0.0)
        if contains_any(result.text, item["answer_contains"]):
            correct += 1
        else:
            wrong.append(item["id"])
        time.sleep(sleep)  # stay inside free-tier rate limits

    answered_anyway, hallucinated = 0, []
    for item in unanswerable:
        result = pipeline.answer(item["question"], temperature=0.0)
        if not result.abstained:
            answered_anyway += 1
            hallucinated.append(item["id"])
        time.sleep(sleep)

    return AnswerResult(
        accuracy=correct / (len(answerable) or 1),
        false_answer_rate=answered_anyway / (len(unanswerable) or 1),
        wrong=wrong,
        hallucinated=hallucinated,
    )


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

SWEEP_GRID = {
    "chunk_size": [400, 900, 1500],
    "chunk_overlap": [0, 150],
    "drop_references": [True, False],
}


def run_sweep(pdf: Path, gold: dict, top_k: int) -> list[dict]:
    rows: list[dict] = []
    keys = list(SWEEP_GRID)
    combos = list(itertools.product(*(SWEEP_GRID[k] for k in keys)))
    scored = sum(1 for q in gold["questions"] if not q.get("unanswerable"))
    print(f"Sweeping {len(combos)} configs over {scored} answerable questions…\n")

    for combo in combos:
        settings = dict(zip(keys, combo))
        config = RAGConfig(top_k=top_k, **settings)
        pipeline, _ = PaperRAG.from_pdf(pdf, config)
        result = score_retrieval(pipeline, gold, top_k)
        rows.append(
            {
                **settings,
                "top_k": top_k,
                "chunks": len(pipeline.chunks),
                "hit_at_k": round(result.hit_at_k, 3),
                "mrr": round(result.mrr, 3),
                "misses": result.misses,
            }
        )
        print(
            f"  size={settings['chunk_size']:<5} overlap={settings['chunk_overlap']:<4} "
            f"drop_refs={str(settings['drop_references']):<5} "
            f"chunks={len(pipeline.chunks):<4} "
            f"hit@{top_k}={result.hit_at_k:.2f}  MRR={result.mrr:.3f}"
        )

    rows.sort(key=lambda r: (-r["hit_at_k"], -r["mrr"]))
    print("\nBest config:")
    best = rows[0]
    print(
        f"  chunk_size={best['chunk_size']}, overlap={best['chunk_overlap']}, "
        f"drop_references={best['drop_references']} "
        f"-> hit@{top_k}={best['hit_at_k']:.2f}, MRR={best['mrr']:.3f}"
    )
    worst = rows[-1]
    lift = best["hit_at_k"] - worst["hit_at_k"]
    print(
        f"  ({lift:+.2f} hit@{top_k} versus the worst config in the grid - "
        "this is the number that belongs in your README)"
    )
    return rows


def run_single(pdf: Path, gold: dict, config: RAGConfig, with_answers: bool) -> dict:
    pipeline, cached = PaperRAG.from_pdf(pdf, config)
    print(f"Indexed {len(pipeline.chunks)} chunks{' (from cache)' if cached else ''}.\n")

    retrieval = score_retrieval(pipeline, gold, config.top_k)
    print("RETRIEVAL")
    print(f"  hit@{config.top_k}      {retrieval.hit_at_k:.2f}  ({retrieval.n} questions)")
    print(f"  MRR         {retrieval.mrr:.3f}")
    if len(retrieval.by_tier) > 1:
        print("\n  by tier:")
        for tier, stats in retrieval.by_tier.items():
            print(
                f"    {tier:<12} hit@{config.top_k}={stats['hit_at_k']:.2f}  "
                f"MRR={stats['mrr']:.3f}  (n={stats['n']})"
            )
    if retrieval.misses:
        print(f"\n  missed      {', '.join(retrieval.misses)}")

    payload = {
        "config": config.to_dict(),
        "chunks": len(pipeline.chunks),
        "retrieval": {
            "hit_at_k": retrieval.hit_at_k,
            "mrr": retrieval.mrr,
            "misses": retrieval.misses,
            "by_tier": retrieval.by_tier,
        },
    }

    if with_answers:
        print("\nANSWERING (calling the LLM, this takes a minute)")
        answers = score_answers(pipeline, gold)
        print(f"  accuracy            {answers.accuracy:.2f}")
        print(f"  false-answer rate   {answers.false_answer_rate:.2f}  (lower is better)")
        if answers.wrong:
            print(f"  wrong               {', '.join(answers.wrong)}")
        if answers.hallucinated:
            print(f"  should have abstained  {', '.join(answers.hallucinated)}")
        payload["answers"] = {
            "accuracy": answers.accuracy,
            "false_answer_rate": answers.false_answer_rate,
            "wrong": answers.wrong,
            "hallucinated": answers.hallucinated,
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=ROOT / "gold_attention.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--keep-references", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="grid search the chunking config")
    parser.add_argument("--answers", action="store_true", help="also score the LLM answers")
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.sweep:
        rows = run_sweep(args.pdf, gold, args.top_k)
        out = RESULTS_DIR / f"sweep-{stamp}.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        config = RAGConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            drop_references=not args.keep_references,
        )
        payload = run_single(args.pdf, gold, config, args.answers)
        out = RESULTS_DIR / f"run-{stamp}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
