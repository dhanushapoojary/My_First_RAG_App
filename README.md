# Chat with your PDF (RAG) — Streamlit App

A simple Retrieval-Augmented Generation (RAG) app to chat with one or more uploaded PDFs. It chunks your PDFs, embeds the chunks into a vector index (FAISS), retrieves top-k relevant chunks, and composes an answer with source citations.

## Features
- Ingestion: PDF parsing, chunking with overlap (chunk_size=800 chars, overlap=150 chars)
- Embeddings: multiple backends
  - SentenceTransformers: `sentence-transformers/all-MiniLM-L6-v2`
  - OpenAI: `text-embedding-3-small` (requires `OPENAI_API_KEY`)
  - FastEmbed: `BAAI/bge-small-en-v1.5` (local, fast, no cloud)
- Vector DB: FAISS inner-product index, normalized vectors
- Retrieval: configurable Top-K (default 5)
- Optional reranker: CrossEncoder (toggle)
- Answering: simple composition from retrieved chunks with citations (doc_id/page/chunk)
- Verification: "Show sources" toggle to inspect grounding

## Quickstart

```bash
# From project root
cd "/home/dhanusha/Dhanusha/RAG App"

# (Optional) Create & activate venv
# python3 -m venv .venv && source .venv/bin/activate

python3 -m pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open the displayed URL (typically http://localhost:8502).

## Usage
1. Upload one or more PDF files.
2. In the sidebar:
   - Select Embedding backend (FastEmbed is recommended for local/offline use).
   - Set Top-K and toggles as desired.
   - If using OpenAI embeddings, paste your `OPENAI_API_KEY`.
3. Click "Build Index".
4. Ask a question in the input box.
5. Toggle "Show sources" to verify the answer with citations.

## Configuration
- Chunking: size=800, overlap=150 (fixed in `app.py`)
- Embedding backends:
  - FastEmbed (local, default if available): `BAAI/bge-small-en-v1.5`
  - SentenceTransformers: `sentence-transformers/all-MiniLM-L6-v2`
  - OpenAI: `text-embedding-3-small`
- Env/Secrets:
  - Environment variable: `OPENAI_API_KEY`
  - Streamlit secrets: create `.streamlit/secrets.toml`
    ```toml
    OPENAI_API_KEY = "sk-..."
    ```

## How it Works
- Ingestion: PDFs parsed with `pypdf`. Each page is split into overlapping character chunks.
- Embeddings: Selected backend encodes chunks to dense vectors (unit-normalized).
- Indexing: Vectors are added to FAISS inner-product index.
- Retrieval: Query embedded and searched for top-k similar chunks; optional reranking (CrossEncoder) can reorder results.
- Answering: A simple, grounded response is composed by concatenating top chunks; citations include `doc_id`, `page`, `chunk`, and similarity score.

## Evaluation / Verification
- Enable "Show sources" to inspect the retrieved chunks used to compose the answer.
- This helps validate grounding and spot hallucinations.

## Troubleshooting
- SentenceTransformers/Transformers import errors (NumPy/TensorFlow conflicts):
  - Prefer the FastEmbed backend in the sidebar to avoid heavy dependencies.
  - Alternatively, use OpenAI backend if you have quota.
- OpenAI 429 insufficient_quota:
  - Switch to FastEmbed backend (local) or update your OpenAI billing/quota.
- FAISS not available:
  - Ensure `faiss-cpu` is installed and importable in your environment.

## File Structure
```
/home/dhanusha/Dhanusha/RAG App/
├─ app.py                      # Streamlit app with ingestion, embeddings, FAISS, retrieval, UI
├─ requirements.txt            # Python dependencies
├─ README.md                   # This file
└─ .streamlit/
   └─ secrets.toml             # (optional) Streamlit secrets (OPENAI_API_KEY)
```

## Notes
- This app uses a simple answer composer to keep dependencies light. If you want LLM-based generation (e.g., OpenAI responses grounded with retrieved chunks), we can wire that up next.
- Embedding normalization and FAISS inner-product are used for cosine-equivalent similarity.
