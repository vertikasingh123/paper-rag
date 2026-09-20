# Research Paper Assistant

Upload a paper, ask questions about it, get answers grounded in the paper's own
text with page-level citations — and a measured retrieval quality number rather
than a vibe.

Embeddings run locally, so indexing costs nothing. Only the answer step calls an
API, and both supported providers have a free tier.

```
PDF  →  text extraction  →  cleanup  →  chunking (page + section kept)
                                            ↓
                              local embeddings (all-MiniLM-L6-v2)
                                            ↓
                                    FAISS (exact cosine)
                                            ↓
   question  →  top-k retrieval  →  LLM answers, citing [n]  →  answer + excerpts
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add ONE key - see below
streamlit run app.py
```

**Getting a free key** (either works, no credit card):

| Provider | Key from | Set in `.env` |
|---|---|---|
| Groq (default) | <https://console.groq.com/keys> | `GROQ_API_KEY` |
| Google Gemini | <https://aistudio.google.com/apikey> | `GEMINI_API_KEY` |

Model IDs get deprecated every few months. If you get a "model not found" error,
run `python -m rag.llm` to print what your provider serves today, then set
`GROQ_MODEL` or `GEMINI_MODEL` in `.env`.

The first run downloads the embedding model (~90 MB) and takes a minute.
After that, indexing a paper takes a few seconds and is cached on disk.

## Verify it works

```bash
python tests/test_pipeline.py
```

23 assertions, no API key, no network, no model download — it swaps in a stub
embedder so the chunking, page provenance, FAISS index and cache are all
exercised offline.

## What's actually interesting here

Most "chat with your PDF" projects stop at *it returned something*. Three
decisions here are the difference between a demo and a tool, and each one is
measurable with the eval harness below.

**1. Answers carry page numbers.** Chunks keep the pages they span and the
section heading above them, so every excerpt is labelled `p. 7 · 5.1 Main
Results`. An answer you can check against the paper in five seconds is worth
more than a fluent one you can't.

**2. The text is repaired before it is embedded.** PDF extraction puts a hard
newline at the end of every visual line and splits words across lines with a
hyphen. Left alone, chunk text stops looking like prose and embedding quality
drops. `rag/ingest.py` rejoins wrapped lines, undoes hyphenation (keeping real
hyphens in `non-Gaussian`), keeps headings on their own line, and drops the
bibliography — which is roughly a fifth of a typical paper and matches queries
on author names and years without ever answering one.

**3. The model is allowed to say no.** If the retrieved excerpts don't contain
the answer, the prompt requires an explicit "the excerpts provided don't cover
this" rather than a confident answer from pretraining. The eval set includes
questions the paper genuinely cannot answer, so this is scored, not assumed.

## Results

Measured on *Attention Is All You Need* (arXiv:1706.03762v7), 48 chunks,
`all-MiniLM-L6-v2` embeddings, `openai/gpt-oss-120b` for answering.

| Metric | Value |
|---|---|
| hit@1 | 0.83 |
| hit@3 | 0.92 |
| hit@5 | 1.00 |
| MRR | 0.896 |
| Answer accuracy | 1.00 |
| False-answer rate (unanswerable questions) | 0.00 |

Config sweep over chunk size, overlap and bibliography handling, 12 configs:

| | hit@3 | MRR |
|---|---|---|
| Worst config (1500 / 150) | 0.83 | 0.694 |
| Best config (900 / 0) | 1.00 | 0.875 |

**Caveat, stated plainly: n = 12.** Every metric moves in steps of 0.083, and
three of the six are pinned at ceiling, so this set can no longer discriminate
between configurations. `evals/gold_attention_v2.json` is a 50-question
replacement — 40 answerable across four difficulty tiers plus 10 unanswerable —
built for exactly that reason. Numbers above will be replaced once it has been
run.

### Things the eval found that inspection would not have

**A scorer bug that was under-reporting accuracy.** Answer accuracy showed 0.92
until the failing case turned out to be correct: the model returned "byte‑pair"
with U+2011 NON-BREAKING HYPHEN, which NFKC does not fold to ASCII. Real
accuracy was 1.00. `normalise()` now folds the full dash and space classes.

**Dropping the bibliography does nothing for retrieval quality.** Identical
hit@k and MRR in all 12 sweep pairings. It does shrink the index 20–25%
(126 → 95 chunks at size 400), so it stays as an efficiency measure, not a
quality one. Reported as the negative result it is.

**Dense retrieval ranks fluent prose above correct formulas.** The positional
encoding passage is mostly LaTeX-derived symbol soup, so its embedding is
diluted and it lands at rank 4 — below three clean, confident, entirely
irrelevant paragraphs about attention. This is the motivating case for adding
BM25: `sinusoid` and `positional` are exact tokens a lexical index would match
immediately.

**Answer-stage numbers are not reproducible at temperature > 0.** The same
question returned "37 000 tokens" on one run and "37 k tokens" on the next.
The harness now pins temperature to 0.0; the app keeps 0.1.

## The eval harness

Two failure modes, scored separately, because they need different fixes.

```bash
# retrieval only - no API key, runs in seconds
python -m evals.run_eval --pdf data/aiayn.pdf --gold evals/gold_attention_v2.json --sweep

# add the LLM answer step
python -m evals.run_eval --pdf data/aiayn.pdf --gold evals/gold_attention_v2.json --answers
```

| Stage | Metric | What a bad number means |
|---|---|---|
| Retrieval | `hit@k` — did any retrieved chunk contain the answer? | The right text never reached the model. No prompt will fix this. |
| Retrieval | `MRR` — how high up was it? | It's being found, but buried under noise. |
| Answering | `accuracy` — did the answer contain the expected fact? | The context was right and the model still missed it. Prompt problem. |
| Answering | `false-answer rate` — did it answer an unanswerable question? | It's filling gaps from pretraining. The worst failure, and invisible without this test. |

`--sweep` grid-searches chunk size, overlap and bibliography handling, ranks the
configs and prints the gap between best and worst. That gap is the number to put
in your README, because it's the part you actually caused.

### Writing a gold set for your own paper

`evals/gold_sample.json` is a worked example against the synthetic paper in
`tests/`, so you can run the harness immediately. To build one for a real paper,
write the question, then paste a distinctive phrase from the passage that
answers it:

```json
{
  "id": "q01",
  "question": "What F1 score does the method achieve?",
  "evidence": ["F1 of 0.87"],
  "answer_contains": ["0.87"]
}
```

Evidence is keyed on text, not page numbers, so the set survives a different PDF
build of the same paper. Add three or four `"unanswerable": true` questions —
plausible things the paper simply doesn't discuss — or you will never find out
how often the model invents an answer.

`evals/gold_attention_v2.json` is the 50-question set for *Attention Is All You
Need* — 10 each of `lookup`, `paraphrase`, `table` and `multihop`, plus 10
unanswerable, four of which are deliberate near-misses (training cost in dollars
when the paper gives FLOPs; batch size in sentences when it's stated in tokens).
Every evidence string is verified verbatim against the chunks this pipeline
produces, so a miss is a retrieval failure and never a typo in the gold set.
`python evals/fetch_paper.py` downloads the PDF it expects.

Thirty-plus questions takes about an hour to write and is the single
highest-leverage hour in the project.

## Layout

```
app.py                     Streamlit UI
rag/
  config.py                every retrieval knob, in one hashable dataclass
  ingest.py                PDF → cleaned, page-tagged, section-labelled chunks
  store.py                 local embeddings + FAISS + on-disk cache
  llm.py                   Groq / Gemini provider layer
  pipeline.py              retrieve → prompt → answer with citations
evals/
  run_eval.py              hit@k, MRR, answer accuracy, false-answer rate
  inspect.py               why did THIS question fail? per-question trace
  gold_sample.json         worked example (runs with no download)
  gold_attention_v2.json   50 questions for arXiv:1706.03762, four tiers
tests/
  make_sample_pdf.py       generates a fake paper with realistic PDF damage
  test_pipeline.py         23 offline assertions
```

The cache key is a hash of the PDF bytes *and* the config, so changing chunk size
rebuilds the index instead of silently serving stale chunks.

## Ideas worth the next few hours

Roughly in order of how much they'd improve the numbers:

1. **Hybrid retrieval** — add BM25 alongside the dense search and merge the two
   rankings. Dense embeddings are bad at exact tokens (`P100`, `d_model`, `0.98`),
   which is exactly what paper questions ask about. Usually the biggest single
   `hit@k` gain available.
2. **Query rewriting** — expand "how big is the dataset" into the vocabulary the
   paper actually uses before embedding it.
3. **A reranker** — retrieve 20 chunks, rerank with a cross-encoder, keep 5.
4. **Multi-paper mode** — index several papers and answer comparison questions,
   with each excerpt labelled by paper.

Measure each one with `run_eval.py` before and after. The before/after number is
the thing worth talking about in an interview.

## Known limitations

- **Scanned PDFs don't work.** No text layer to extract; run OCR first.
- **`drop_references` also drops appendices.** It truncates at the bibliography,
  so anything after it — including the attention visualisations on pages 13–15
  of the sample paper — goes with it. Should resume after the reference block
  rather than cutting to the end.
- **Dehyphenation is a heuristic.** `source-\ntarget` becomes `sourcetarget`.
  Harmless for retrieval, visible in quoted excerpts.
- **pypdf drops spaces at some glyph boundaries** (`differentways`,
  `theoutput`). Worth benchmarking PyMuPDF as a replacement.
- **Tables and figures extract as loose text**, so single-cell lookups are
  unreliable — which is why the v2 gold set has a `table` tier to measure it.
- **One paper at a time.** Cross-paper questions need a document ID in the
  metadata and a per-document filter at search time.
