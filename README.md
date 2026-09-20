# Research Paper Assistant

**Ask questions about research papers and get answers grounded in the paper itself.**

Research Paper Assistant is an LLM-powered application that allows users to upload a research paper in PDF format, ask questions about its contents, and receive answers supported by relevant page-level excerpts.

Unlike a basic “chat with your PDF” application, this project includes an evaluation framework to measure how effectively the system retrieves relevant information and avoids making unsupported claims.

## How It Works

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

```text
Upload PDF
    ↓
Extract and clean the text
    ↓
Split the paper into smaller sections
    ↓
Generate local text embeddings
    ↓
Store embeddings in a FAISS vector index
    ↓
User asks a question
    ↓
Retrieve the most relevant sections
    ↓
LLM generates an answer using the retrieved text
    ↓
Display the answer with page references and source excerpts
```

### Key Features

* **Question answering over PDFs:** Ask natural-language questions about an uploaded research paper.
* **Page-level citations:** Answers include references to the pages and sections supporting them.
* **Local embeddings:** Text embeddings are generated locally, so document indexing does not require an API key.
* **Cached indexing:** Previously indexed documents can be reused without repeating the entire process.
* **Abstention from unsupported answers:** If the retrieved content does not contain the answer, the system is instructed to acknowledge that limitation instead of guessing.
* **Retrieval evaluation:** Measure retrieval quality using metrics such as `hit@k` and Mean Reciprocal Rank (MRR).
* **Offline testing:** Validate text processing, chunking, metadata preservation, FAISS indexing, and caching without requiring an API key.

## Why This Project Matters

A fluent answer is not necessarily a correct answer. In research-oriented applications, it is important to know:

1. Whether the system retrieved the right information.
2. Whether the language model used that information correctly.
3. Whether the system avoided inventing answers when the paper did not provide the required information.

This project evaluates these stages separately rather than judging the application only by whether it produces a plausible response.

## Technology Stack

* **Frontend:** Streamlit
* **Document processing:** PDF text extraction and cleaning
* **Embeddings:** `all-MiniLM-L6-v2`
* **Vector search:** FAISS with cosine similarity
* **LLM providers:** Groq and Google Gemini
* **Evaluation:** Retrieval accuracy, MRR, answer accuracy, and false-answer rate
* **Testing:** Offline automated test suite

## Quickstart

### 1. Set up the environment

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure an LLM provider

Copy the example environment file:

```bash
cp .env.example .env
```

Add one API key to `.env`:

| Provider      | API key          |
| ------------- | ---------------- |
| Groq          | `GROQ_API_KEY`   |
| Google Gemini | `GEMINI_API_KEY` |

The embedding model runs locally. Only the answer-generation step requires an LLM API.

### 3. Launch the application

```bash
streamlit run app.py
```

The first run downloads the embedding model. Subsequent document indexing operations are cached on disk.

## Evaluation

The project includes an evaluation harness that measures retrieval and answer-generation quality separately.

| Metric            | What it measures                                                      |
| ----------------- | --------------------------------------------------------------------- |
| `hit@k`           | Whether a relevant passage appears among the top `k` retrieved chunks |
| MRR               | How highly the first relevant passage is ranked                       |
| Answer accuracy   | Whether the generated answer contains the expected information        |
| False-answer rate | How often the system answers questions that the paper cannot answer   |

Run the offline test suite:

```bash
python tests/test_pipeline.py
```

Run retrieval evaluation:

```bash
python -m evals.run_eval \
    --pdf data/aiayn.pdf \
    --gold evals/gold_attention_v2.json \
    --sweep
```

Run evaluation including the LLM answer-generation stage:

```bash
python -m evals.run_eval \
    --pdf data/aiayn.pdf \
    --gold evals/gold_attention_v2.json \
    --answers
```

## Results

An initial evaluation was conducted using *Attention Is All You Need* with 48 chunks, the `all-MiniLM-L6-v2` embedding model, and `openai/gpt-oss-120b` for answer generation.

| Metric            | Result |
| ----------------- | -----: |
| Hit@1             |   0.83 |
| Hit@3             |   0.92 |
| Hit@5             |   1.00 |
| MRR               |  0.896 |
| Answer accuracy   |   1.00 |
| False-answer rate |   0.00 |

**Evaluation caveat:** The initial evaluation contained only 12 questions. These results should therefore be treated as preliminary rather than definitive evidence of general performance. A larger 50-question evaluation set is included in the repository for further testing.

## Project Structure

```text
app.py                     Streamlit interface

rag/
├── config.py              Configuration management
├── ingest.py              PDF processing and text chunking
├── store.py               Embeddings, FAISS, and caching
├── llm.py                 LLM provider integration
└── pipeline.py            Retrieval and answer generation

evals/
├── run_eval.py            Evaluation metrics and configuration sweeps
├── inspect.py             Per-question failure analysis
├── gold_sample.json       Example evaluation set
└── gold_attention_v2.json Larger evaluation set

tests/
├── make_sample_pdf.py     Generates a sample PDF
└── test_pipeline.py       Offline test suite
```

## Current Limitations

* Scanned PDFs require OCR before text can be extracted.
* Tables and figures may not be extracted reliably.
* The system currently supports one paper at a time.
* PDF text extraction may introduce spacing and hyphenation errors.
* Cross-paper comparison is not currently supported.

## Future Improvements

* Add hybrid retrieval using dense embeddings and BM25 keyword search.
* Add query rewriting for technical terminology.
* Introduce a reranking model to improve retrieval precision.
* Support multi-paper comparison and document-level filtering.

## License

See the `LICENSE` file for licensing information.
