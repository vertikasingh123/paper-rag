"""Streamlit front end for the paper assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from rag.config import RAGConfig
from rag.llm import LLMError
from rag.pipeline import PaperRAG

st.set_page_config(page_title="Paper Assistant", page_icon="📄", layout="wide")

SAMPLE_QUESTIONS = [
    ("Problem", "What problem does this paper set out to solve?"),
    ("Data", "What data did they use, and how large is it?"),
    ("Result", "What is the main result, with specific numbers?"),
    ("Baselines", "What baselines or prior methods do they compare against?"),
    ("Limitations", "What limitations do the authors acknowledge?"),
]


# --------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def build_pipeline(pdf_bytes: bytes, config_dict: tuple) -> tuple[PaperRAG, int]:
    """Cached per (file contents, config) so re-asking never re-indexes."""
    config = RAGConfig(**dict(config_dict))
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(pdf_bytes)
        temp_path = Path(handle.name)
    try:
        pipeline, _ = PaperRAG.from_pdf(temp_path, config)
        return pipeline, len(pipeline.chunks)
    finally:
        temp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Retrieval settings")
    st.caption("Changing any of these rebuilds the index.")
    chunk_size = st.slider("Chunk size (characters)", 300, 2000, 900, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Chunks retrieved per question", 1, 12, 5)
    drop_references = st.checkbox(
        "Drop the bibliography",
        value=True,
        help="References are ~20% of a paper and match on author names and "
        "years without ever answering a question.",
    )
    dehyphenate = st.checkbox(
        "Repair hyphenated line breaks",
        value=True,
        help="Turns 'hyphen-\\nation' back into 'hyphenation' before embedding.",
    )

    st.divider()
    st.caption(
        "Embeddings run locally (all-MiniLM-L6-v2). Only the answer step "
        "calls an API."
    )

config = RAGConfig(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    top_k=top_k,
    drop_references=drop_references,
    dehyphenate=dehyphenate,
)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

st.title("📄 Research Paper Assistant")
st.caption("Ask questions about a paper and get answers grounded in its own text.")

uploaded = st.file_uploader("Upload a paper (PDF)", type="pdf")

if not uploaded:
    st.info("Upload a PDF to get started. Any arXiv paper works well as a first test.")
    st.stop()

pdf_bytes = uploaded.getvalue()
config_key = tuple(sorted(config.to_dict().items()))

with st.spinner("Indexing the paper… (first run downloads the embedding model)"):
    try:
        pipeline, n_chunks = build_pipeline(pdf_bytes, config_key)
    except ValueError as exc:
        st.error(
            f"{exc}\n\nThis usually means the PDF is a scan with no text layer. "
            "Run it through OCR first."
        )
        st.stop()

pages = sorted({page for chunk in pipeline.chunks for page in chunk.pages})
col_a, col_b, col_c = st.columns(3)
col_a.metric("Chunks indexed", n_chunks)
col_b.metric("Pages covered", len(pages))
col_c.metric("Retrieved per question", config.top_k)

if "history" not in st.session_state:
    st.session_state.history = []

st.write("**Try one of these:**")
cols = st.columns(len(SAMPLE_QUESTIONS))
preset = None
for col, (label, question) in zip(cols, SAMPLE_QUESTIONS):
    if col.button(label, help=question, use_container_width=True):
        preset = question

asked = st.chat_input("Ask a question about the paper…") or preset

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for source in entry["sources"]:
                st.markdown(
                    f"**[{source['rank']}]** {source['label']} "
                    f"· score {source['score']:.3f}"
                )
                st.caption(source["text"])

if asked:
    with st.chat_message("user"):
        st.write(asked)
    with st.chat_message("assistant"):
        with st.spinner("Reading the relevant sections…"):
            try:
                result = pipeline.answer(asked)
            except LLMError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # network, bad model id, rate limit
                st.error(
                    f"The answer step failed: {exc}\n\n"
                    "If this mentions a model that does not exist, run "
                    "`python -m rag.llm` to list the models your provider "
                    "currently serves, then set it in .env."
                )
                st.stop()

        if result.abstained:
            st.warning(result.text)
        else:
            st.markdown(result.text)

        with st.expander(f"Sources ({len(result.sources)})", expanded=True):
            for source in result.sources:
                st.markdown(
                    f"**[{source.rank}]** {source.label} · score {source.score:.3f}"
                )
                st.caption(source.chunk.text)

    st.session_state.history.append(
        {
            "question": asked,
            "answer": result.text,
            "sources": [
                {
                    "rank": s.rank,
                    "label": s.label,
                    "score": s.score,
                    "text": s.chunk.text,
                }
                for s in result.sources
            ],
        }
    )
