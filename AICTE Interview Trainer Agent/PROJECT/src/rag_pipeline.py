"""
rag_pipeline.py – RAG (Retrieval-Augmented Generation) pipeline.

Handles:
  - Document ingestion from the knowledge base directory
  - Text chunking and embedding via sentence-transformers
  - Vector storage and retrieval via ChromaDB
  - Context assembly for IBM Granite prompts
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed. Using in-memory mock retrieval.")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using keyword search fallback.")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

from src.config import config


# ──────────────────────────────────────────────────────────────────────────────
# Text Splitter
# ──────────────────────────────────────────────────────────────────────────────

class TextSplitter:
    """Simple recursive character text splitter."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try to break at sentence boundary
            if end < text_len:
                for sep in ["\n\n", "\n", ". ", " "]:
                    boundary = text.rfind(sep, start, end)
                    if boundary > start:
                        end = boundary + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Document Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return ""


def load_pdf_file(path: Path) -> str:
    if not PYPDF_AVAILABLE:
        logger.warning(f"pypdf not installed; skipping {path}")
        return ""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        logger.error(f"Failed to load PDF {path}: {e}")
        return ""


def load_docx_file(path: Path) -> str:
    if not DOCX_AVAILABLE:
        logger.warning(f"python-docx not installed; skipping {path}")
        return ""
    try:
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"Failed to load DOCX {path}: {e}")
        return ""


def load_json_file(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return "\n".join(str(item) for item in data)
        elif isinstance(data, dict):
            return json.dumps(data, indent=2)
        return str(data)
    except Exception as e:
        logger.error(f"Failed to load JSON {path}: {e}")
        return ""


def load_document(path: Path) -> str:
    """Dispatch to the appropriate loader based on file extension."""
    ext = path.suffix.lower()
    dispatch = {
        ".txt": load_text_file,
        ".md": load_text_file,
        ".pdf": load_pdf_file,
        ".docx": load_docx_file,
        ".json": load_json_file,
    }
    loader = dispatch.get(ext, load_text_file)
    return loader(path)


# ──────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline:
      ingest documents → embed → store in ChromaDB → retrieve on query.
    """

    def __init__(self):
        self.cfg_vs = config.vectorstore
        self.cfg_kb = config.knowledge_base
        self.splitter = TextSplitter(
            chunk_size=self.cfg_vs.chunk_size,
            chunk_overlap=self.cfg_vs.chunk_overlap,
        )

        self._client: Optional[object] = None
        self._collection: Optional[object] = None
        self._embedder: Optional[object] = None
        self._in_memory_docs: List[Dict] = []  # fallback

        self._setup_embedder()
        self._setup_vectorstore()

    # ── Setup ────────────────────────────────────────────────────────────────

    def _setup_embedder(self):
        if ST_AVAILABLE:
            try:
                self._embedder = SentenceTransformer(self.cfg_vs.embedding_model)
                logger.info(f"Embedding model loaded: {self.cfg_vs.embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedder = None
        else:
            self._embedder = None

    def _setup_vectorstore(self):
        if not CHROMA_AVAILABLE:
            logger.warning("ChromaDB unavailable. Using in-memory keyword search.")
            return

        try:
            persist_dir = Path(self.cfg_vs.persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.cfg_vs.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB initialized at {persist_dir}")
        except Exception as e:
            logger.error(f"ChromaDB setup failed: {e}")
            self._collection = None

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest_knowledge_base(self) -> int:
        """
        Scan the knowledge base directory and ingest all documents.
        Returns the total number of chunks added.
        """
        kb_dir = Path(self.cfg_kb.base_dir)
        if not kb_dir.exists():
            logger.warning(f"Knowledge base directory not found: {kb_dir}")
            return 0

        supported = {".txt", ".md", ".pdf", ".docx", ".json"}
        total_chunks = 0

        for doc_path in kb_dir.rglob("*"):
            if doc_path.is_file() and doc_path.suffix.lower() in supported:
                chunks_added = self._ingest_file(doc_path)
                total_chunks += chunks_added
                logger.info(f"  Ingested {doc_path.name}: {chunks_added} chunks")

        logger.info(f"Knowledge base ingestion complete: {total_chunks} total chunks")
        return total_chunks

    def ingest_resume(self, file_path: str) -> int:
        """Ingest a user-provided resume file into the vector store."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Resume file not found: {file_path}")
            return 0
        return self._ingest_file(path, source_type="resume")

    def ingest_text(self, text: str, doc_id: str, metadata: Optional[Dict] = None) -> int:
        """Directly ingest raw text (e.g., pasted resume content)."""
        chunks = self.splitter.split(text)
        if not chunks:
            return 0
        meta = metadata or {}
        meta["source"] = doc_id
        meta["type"] = meta.get("type", "user_input")
        self._store_chunks(chunks, meta)
        return len(chunks)

    def _ingest_file(self, path: Path, source_type: str = "knowledge_base") -> int:
        text = load_document(path)
        if not text.strip():
            return 0

        chunks = self.splitter.split(text)
        if not chunks:
            return 0

        metadata = {
            "source": path.name,
            "type": source_type,
            "path": str(path),
        }
        self._store_chunks(chunks, metadata)
        return len(chunks)

    def _store_chunks(self, chunks: List[str], metadata: Dict):
        """Store chunks in ChromaDB or in-memory fallback."""
        import hashlib

        if self._collection is not None and self._embedder is not None:
            try:
                ids = [
                    hashlib.md5(f"{metadata['source']}_{i}_{c[:30]}".encode()).hexdigest()
                    for i, c in enumerate(chunks)
                ]
                embeddings = self._embedder.encode(chunks, show_progress_bar=False).tolist()
                metas = [metadata.copy() for _ in chunks]

                # ChromaDB upsert (handles duplicates gracefully)
                self._collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=metas,
                )
                return
            except Exception as e:
                logger.warning(f"ChromaDB store failed: {e}. Falling back to memory.")

        # In-memory fallback
        for chunk in chunks:
            self._in_memory_docs.append({"text": chunk, "metadata": metadata.copy()})

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        job_role: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve the most relevant document chunks for a given query.
        Optionally filter by job_role metadata.
        """
        k = top_k or self.cfg_vs.top_k_results

        if self._collection is not None and self._embedder is not None:
            return self._chroma_retrieve(query, k)
        return self._keyword_retrieve(query, k)

    def _chroma_retrieve(self, query: str, top_k: int) -> List[Dict]:
        try:
            query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, max(1, self._collection.count())),
            )
            docs = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                meta = results.get("metadatas", [[]])[0][i] if results.get("metadatas") else {}
                dist = results.get("distances", [[]])[0][i] if results.get("distances") else 0.0
                docs.append({
                    "text": doc,
                    "metadata": meta,
                    "score": round(1.0 - dist, 4),
                })
            return docs
        except Exception as e:
            logger.error(f"ChromaDB retrieval error: {e}")
            return self._keyword_retrieve(query, top_k)

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """Simple TF-IDF-style keyword overlap retrieval."""
        query_words = set(query.lower().split())
        scored: List[Tuple[float, Dict]] = []

        docs = self._in_memory_docs or FALLBACK_KNOWLEDGE

        for doc in docs:
            text = doc.get("text", "")
            doc_words = set(text.lower().split())
            overlap = len(query_words & doc_words)
            score = overlap / max(len(query_words), 1)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": d["text"], "metadata": d.get("metadata", {}), "score": round(s, 4)}
            for s, d in scored[:top_k]
        ]

    def build_context(
        self,
        query: str,
        job_role: Optional[str] = None,
        max_chars: int = 3000,
    ) -> str:
        """
        Retrieve relevant chunks and concatenate into a single context string
        suitable for inclusion in an IBM Granite prompt.
        """
        results = self.retrieve(query, job_role=job_role)
        if not results:
            return "No specific context retrieved. Using general interview knowledge."

        parts = []
        total = 0
        for r in results:
            text = r["text"]
            source = r["metadata"].get("source", "unknown")
            snippet = f"[Source: {source}]\n{text}"
            if total + len(snippet) > max_chars:
                break
            parts.append(snippet)
            total += len(snippet)

        return "\n\n---\n\n".join(parts)

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        count = 0
        if self._collection:
            try:
                count = self._collection.count()
            except Exception:
                pass
        return {
            "chroma_available": CHROMA_AVAILABLE,
            "embedder_available": ST_AVAILABLE,
            "collection_count": count,
            "in_memory_docs": len(self._in_memory_docs),
            "collection_name": self.cfg_vs.collection_name,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Built-in Fallback Knowledge (used when no knowledge base files are present)
# ──────────────────────────────────────────────────────────────────────────────

FALLBACK_KNOWLEDGE = [
    {
        "text": (
            "Software Engineer Interview Tips: Focus on data structures, algorithms, "
            "system design, and object-oriented principles. Practice STAR behavioral "
            "questions. Research the company tech stack thoroughly."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "Data Scientist Interview: Prepare for questions on machine learning "
            "algorithms, statistical concepts, Python/R programming, SQL, and "
            "model evaluation metrics. Be ready to discuss past projects with business impact."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "Product Manager Interview: Demonstrate product thinking, user empathy, "
            "prioritization frameworks (RICE, MoSCoW), roadmap planning, and cross-functional "
            "leadership. Prepare metrics-driven success stories."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "Behavioral Interview - STAR Method: Structure answers as Situation (context), "
            "Task (your responsibility), Action (steps you took), Result (outcome with metrics). "
            "Always quantify results when possible."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "HR Interview Common Questions: Tell me about yourself (structured pitch), "
            "Why this company? (research-backed), Where do you see yourself in 5 years? "
            "(aligned with role), Salary expectations (market-researched range), "
            "Questions for the interviewer (show engagement)."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "DevOps Engineer Interview: Linux/Unix commands, CI/CD pipelines (Jenkins, "
            "GitLab CI, GitHub Actions), containerization (Docker, Kubernetes), "
            "Infrastructure as Code (Terraform, Ansible), monitoring (Prometheus, Grafana), "
            "cloud platforms (AWS, Azure, IBM Cloud)."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "Interview Confidence Tips: Practice answers aloud, not silently. "
            "Use power posing before interviews. Prepare 3 strong stories that showcase "
            "leadership, problem-solving, and collaboration. Silence is okay — take a "
            "moment to think before answering complex questions."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
    {
        "text": (
            "IBM Cloud and watsonx.ai: IBM's enterprise AI platform offers foundation models "
            "including IBM Granite for code and language tasks. Key services: watsonx.ai "
            "(model studio), watsonx.data (data lakehouse), watsonx.governance (AI lifecycle). "
            "Candidates for AI/ML roles should understand IBM's responsible AI principles."
        ),
        "metadata": {"source": "builtin", "type": "knowledge_base"},
    },
]
