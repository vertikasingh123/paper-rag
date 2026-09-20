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

## The eval harness

Two failure modes, scored separately, because they need different fixes.

```bash
# retrieval only - no API key, runs in seconds
python -m evals.run_eval --pdf data/attention.pdf --sweep

# add the LLM answer step
python -m evals.run_eval --pdf data/attention.pdf --answers
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

`evals/gold_attention.json` is a 15-question set for *Attention Is All You Need*;
`python evals/fetch_paper.py` downloads the PDF it expects.

Thirty questions takes about an hour to write and is the single highest-leverage
hour in the project.

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
  gold_sample.json         worked example (runs with no download)
  gold_attention.json      15 questions for arXiv:1706.03762
tests/
  make_sample_pdf.py       generates a fake paper with realistic PDF damage
  test_pipeline.py         23 offline assertions
```

The cache key is a hash of the PDF bytes *and* the config, so changing chunk size
rebuilds the index instead of silently serving stale chunks.

## Known limitations

- **Scanned PDFs don't work.** There's no text layer to extract; run OCR first.
- **Dehyphenation is a heuristic.** `held-\nout` becomes `heldout`. Harmless for
  retrieval, occasionally visible in a quoted excerpt.
- **Tables and figures are extracted as loose text**, so questions about a
  specific cell in a table are unreliable.
- **One paper at a time.** Cross-paper questions need a document ID in the
  metadata and a per-document filter at search time.

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

## Resume line

Once you've run the sweep on a real paper, write the line with your own numbers:

> Built a retrieval-augmented QA system over research PDFs (Python, Streamlit,
> FAISS, sentence-transformers). Wrote a 15-question gold set with deliberately
> unanswerable cases; grid-searched the chunking config to raise retrieval hit@5
> from **0.XX to 0.YY**, and cut the false-answer rate on unanswerable questions
> from **0.XX to 0.YY** with an abstention prompt.
