import os
# Disable optional backends to avoid heavy/fragile imports (e.g., TF requiring specific NumPy)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import hashlib
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
from pypdf import PdfReader

try:
	import faiss  # type: ignore
except Exception as e:
	faiss = None

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

try:
	from openai import OpenAI  # type: ignore
except Exception:
	OpenAI = None  # type: ignore

try:
	from fastembed import TextEmbedding
except Exception:
	TextEmbedding = None  # type: ignore


# ------------------------- Config -------------------------
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 150  # characters
TOP_K_DEFAULT = 5


@dataclass
class Chunk:
	text: str
	metadata: Dict[str, Any]
	vector: Optional[np.ndarray] = None


# ------------------------- Utilities -------------------------
def compute_sha1(content: bytes) -> str:
	sha1 = hashlib.sha1()
	sha1.update(content)
	return sha1.hexdigest()


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
	chunks: List[str] = []
	start = 0
	text_length = len(text)
	while start < text_length:
		end = min(start + chunk_size, text_length)
		chunks.append(text[start:end])
		if end == text_length:
			break
		start = end - overlap
		if start < 0:
			start = 0
	return chunks


# ------------------------- Ingestion -------------------------
def read_pdf_return_pages(file_path: str) -> List[str]:
	reader = PdfReader(file_path)
	pages: List[str] = []
	for page in reader.pages:
		pages.append(page.extract_text() or "")
	return pages


def ingest_pdfs(file_paths: List[str], doc_ids: List[str]) -> List[Chunk]:
	all_chunks: List[Chunk] = []
	for file_path, doc_id in zip(file_paths, doc_ids):
		pages = read_pdf_return_pages(file_path)
		for page_num, page_text in enumerate(pages, start=1):
			if not page_text:
				continue
			pieces = split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
			for chunk_index, piece in enumerate(pieces):
				all_chunks.append(
					Chunk(
						text=piece,
						metadata={
							"doc_id": doc_id,
							"page": page_num,
							"chunk": chunk_index,
						},
					)
				)
	return all_chunks


# ------------------------- Embeddings with cache -------------------------
class EmbeddingService:
	def __init__(self, backend: str, local_model_name: str = DEFAULT_EMBED_MODEL, openai_model_name: str = OPENAI_EMBED_MODEL):
		self.backend = backend
		self.local_model_name = local_model_name
		self.openai_model_name = openai_model_name
		self.cache: Dict[str, np.ndarray] = {}
		self.local_model = None
		self.openai_client = None
		self.fastembed_model = None
		if backend == "SentenceTransformers":
			try:
				from sentence_transformers import SentenceTransformer  # lazy import
				self.local_model = SentenceTransformer(local_model_name)
			except Exception as e:
				raise RuntimeError(f"Failed to load SentenceTransformers model: {e}")
		elif backend == "OpenAI":
			if OpenAI is None:
				raise RuntimeError("openai package not available")
			api_key = os.environ.get("OPENAI_API_KEY")
			if not api_key:
				raise RuntimeError("OPENAI_API_KEY not set")
			self.openai_client = OpenAI()
		elif backend == "FastEmbed":
			if TextEmbedding is None:
				raise RuntimeError("fastembed not available")
			# Default small, fast local model
			self.fastembed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
		else:
			raise ValueError("Unknown embedding backend")

	def _cache_key(self, text: str) -> str:
		return hashlib.md5((self.backend + "::" + text).encode("utf-8")).hexdigest()

	def embed_texts(self, texts: List[str]) -> np.ndarray:
		embeddings: List[np.ndarray] = []
		for t in texts:
			key = self._cache_key(t)
			if key in self.cache:
				emb = self.cache[key]
			else:
				if self.backend == "SentenceTransformers":
					emb = self.local_model.encode(t, normalize_embeddings=True)  # type: ignore
				elif self.backend == "OpenAI":
					resp = self.openai_client.embeddings.create(model=self.openai_model_name, input=t)  # type: ignore
					emb = np.array(resp.data[0].embedding, dtype=np.float32)
					norm = np.linalg.norm(emb) + 1e-12
					emb = emb / norm
				elif self.backend == "FastEmbed":
					# fastembed returns generator of embeddings
					arr = list(self.fastembed_model.embed([t]))[0]  # type: ignore
					emb = np.array(arr, dtype=np.float32)
					norm = np.linalg.norm(emb) + 1e-12
					emb = emb / norm
				else:
					raise ValueError("Unknown backend")
				self.cache[key] = emb
			embeddings.append(emb)
		return np.vstack(embeddings).astype(np.float32)


# ------------------------- Vector Store (FAISS) -------------------------
class FaissStore:
	def __init__(self, dim: int):
		if faiss is None:
			raise RuntimeError("faiss is not available. Please install faiss-cpu.")
		self.index = faiss.IndexFlatIP(dim)
		self.metadatas: List[Dict[str, Any]] = []
		self.texts: List[str] = []

	def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]], texts: List[str]):
		faiss.normalize_L2(vectors)
		self.index.add(vectors.astype(np.float32))
		self.metadatas.extend(metadatas)
		self.texts.extend(texts)

	def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
		faiss.normalize_L2(query_vec)
		scores, indices = self.index.search(query_vec.astype(np.float32), top_k)
		results: List[Tuple[int, float]] = []
		for i, score in zip(indices[0], scores[0]):
			if i == -1:
				continue
			results.append((int(i), float(score)))
		return results

	def get(self, idx: int) -> Tuple[str, Dict[str, Any]]:
		return self.texts[idx], self.metadatas[idx]


# ------------------------- Reranker (optional) -------------------------
class Reranker:
	def __init__(self, model_name: str = DEFAULT_RERANKER):
		try:
			from sentence_transformers import CrossEncoder  # lazy import
			self.model = CrossEncoder(model_name)
		except Exception:
			self.model = None

	def rerank(self, query: str, candidates: List[Tuple[str, Dict[str, Any], float]], top_k: int) -> List[Tuple[str, Dict[str, Any], float]]:
		if not self.model or not candidates:
			return candidates[:top_k]
		pairs = [(query, text) for text, _, _ in candidates]
		scores = self.model.predict(pairs)
		scored = [(text, meta, float(score)) for (text, meta, _), score in zip(candidates, scores)]
		scored.sort(key=lambda x: x[2], reverse=True)
		return scored[:top_k]


# ------------------------- QA / Answer Composer -------------------------
def compose_answer(query: str, contexts: List[Tuple[str, Dict[str, Any], float]]) -> Tuple[str, List[Dict[str, Any]]]:
	# Simple extractive-style composition without external LLM
	citations: List[Dict[str, Any]] = []
	for text, meta, score in contexts:
		citations.append({"doc_id": meta.get("doc_id"), "page": meta.get("page"), "chunk": meta.get("chunk"), "score": round(score, 4)})

	joined_context = "\n\n".join([c[0] for c in contexts])
	answer = (
		"Based on the uploaded PDFs, here is a consolidated answer.\n\n"
		"Context snippets considered (top results):\n"
		f"{joined_context}\n\n"
		"Citations indicate source doc_id, page, and chunk."
	)
	return answer, citations


# ------------------------- Streamlit UI -------------------------
def main():
	st.set_page_config(page_title="Chat with your PDF (RAG)", page_icon="📄")
	st.title("📄 Chat with your PDF")
	st.caption("Chunk size: 800 chars, overlap: 150 chars. Vector DB: FAISS. Embeddings: all-MiniLM-L6-v2 or OpenAI.")

	# Load API key from env or Streamlit secrets
	default_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""

	with st.sidebar:
		st.header("Settings")
		default_backend_index = 1 if default_api_key else 2 if TextEmbedding is not None else 1
		emb_backend = st.selectbox("Embedding backend", ["SentenceTransformers", "OpenAI", "FastEmbed"], index=default_backend_index)
		top_k = st.slider("Top-K", 1, 20, TOP_K_DEFAULT)
		use_reranker = st.checkbox("Use reranker (CrossEncoder)", value=False)
		show_sources = st.checkbox("Show sources", value=True)
		st.divider()
		api_key_input = st.text_input("OPENAI_API_KEY (for OpenAI backend)", value=default_api_key or "", type="password")
		if api_key_input:
			os.environ["OPENAI_API_KEY"] = api_key_input
			st.session_state["OPENAI_API_KEY"] = api_key_input
		st.caption("Upload PDFs and click 'Build Index'.")

	uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
	query = st.text_input("Ask a question about the PDFs")
	build = st.button("Build Index")

	if "index" not in st.session_state:
		st.session_state.index = None
		st.session_state.texts = []
		st.session_state.metadatas = []
		st.session_state.embed_backend = "SentenceTransformers"

	if build and uploaded_files:
		with st.spinner("Processing PDFs and building index..."):
			# Save to temp files for pypdf
			temp_paths: List[str] = []
			doc_ids: List[str] = []
			for f in uploaded_files:
				data = f.read()
				sha = compute_sha1(data)
				doc_ids.append(sha[:10])
				tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
				tmp.write(data)
				tmp.flush()
				temp_paths.append(tmp.name)

			chunks = ingest_pdfs(temp_paths, doc_ids)
			try:
				embedder = EmbeddingService(backend=emb_backend)
				vectors = embedder.embed_texts([c.text for c in chunks]).astype(np.float32)
			except Exception as e:
				st.error(f"Embedding backend failed: {e}")
				return

			store = FaissStore(dim=vectors.shape[1])
			store.add(vectors, [c.metadata for c in chunks], [c.text for c in chunks])

			st.session_state.index = store
			st.session_state.texts = [c.text for c in chunks]
			st.session_state.metadatas = [c.metadata for c in chunks]
			st.session_state.embed_backend = emb_backend

			st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDFs.")

		# cleanup temp files
		for p in temp_paths:
			try:
				os.unlink(p)
			except Exception:
				pass

	if query and st.session_state.index is not None:
		with st.spinner("Retrieving..."):
			try:
				embedder = EmbeddingService(backend=st.session_state.embed_backend)
				q_vec = embedder.embed_texts([query])
			except Exception as e:
				st.error(f"Embedding backend failed: {e}")
				return
			results = st.session_state.index.search(q_vec, top_k=top_k)
			candidates: List[Tuple[str, Dict[str, Any], float]] = []
			for idx, score in results:
				text, meta = st.session_state.index.get(idx)
				candidates.append((text, meta, score))

			if use_reranker:
				reranker = Reranker()
				candidates = reranker.rerank(query, candidates, top_k)

			answer, citations = compose_answer(query, candidates)

			st.subheader("Answer")
			st.write(answer)

			if show_sources:
				st.subheader("Sources")
				for i, c in enumerate(citations, start=1):
					st.write(f"{i}. doc_id={c['doc_id']} page={c['page']} chunk={c['chunk']} score={c['score']}")
					with st.expander("Show chunk text"):
						st.write(candidates[i-1][0])

	else:
		st.info("Upload PDFs, build the index, and ask a question.")


if __name__ == "__main__":
	main()
