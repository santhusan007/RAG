import os
import re
import tempfile
from typing import List

import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_pdf(file_bytes: bytes) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    docs: List[Document] = []
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    except Exception:
        docs = []

    if not docs or sum(len(d.page_content or "") for d in docs) < 50:
        pages: List[Document] = []
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    txt = page.extract_text() or ""
                    txt = clean_text(txt)
                    if txt:
                        pages.append(Document(page_content=txt, metadata={"page": i}))
        except Exception:
            pages = []
        docs = pages

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.page_content = clean_text(c.page_content)
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def get_or_build_vectorstore(chunks: List[Document], file_id: str) -> Chroma:
    base_dir = os.path.join(".rag_index", file_id)
    os.makedirs(base_dir, exist_ok=True)
    embeddings = get_embeddings()
    try:
        vs = Chroma(persist_directory=base_dir, embedding_function=embeddings)
        # If empty collection (fresh directory), populate
        try:
            count = 0
            try:
                # Newer chroma exposes count via underlying collection
                count = vs._collection.count()  # type: ignore[attr-defined]
            except Exception:
                count = 0
            if count == 0:
                vs.add_documents(chunks)
        except Exception:
            pass
        return vs
    except Exception:
        # Fresh build
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=base_dir)
        return vs


def search_documents(vectorstore: Chroma, query: str, k: int = 6, method: str = "auto", fetch_k: int | None = None) -> List[Document]:
    """Search documents with optional method control.

    method: 'similarity' | 'mmr' | 'auto'
    - similarity: fastest approximate relevant chunks
    - mmr: diverse results (slower)
    - auto: defaults to similarity for speed
    """
    try:
        if method == "mmr":
            fk = fetch_k if fetch_k is not None else max(k + 2, int(1.5 * k))
            return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fk)
        # default: similarity for performance
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        # final fallback
        return vectorstore.similarity_search(query, k=k)


def smart_search(
    vectorstore: Chroma,
    query: str,
    k: int = 6,
    method: str = "auto",
    fetch_k: int | None = None,
    chunks: List[Document] | None = None,
) -> List[Document]:
    """Keyword-boosted retrieval: exact phrase/keyword hits from chunks first, then embedding search.

    This helps when the user references a specific section title (e.g., "Target candidate description").
    """
    candidates: List[Document] = []
    seen = set()
    qnorm = re.sub(r"\s+", " ", (query or "").strip().lower())
    m = re.search(r'"([^"]{3,120})"', qnorm)
    phrase = m.group(1).strip() if m else (qnorm if 3 <= len(qnorm) <= 80 else "")
    if chunks and phrase:
        for d in chunks:
            text = (d.page_content or "").lower()
            if phrase in text:
                key = (text[:128], d.metadata.get("page"))
                if key not in seen:
                    seen.add(key)
                    candidates.append(d)
                    if len(candidates) >= k:
                        break
    if len(candidates) < k:
        retrieved = search_documents(vectorstore, query, k=k, method=method, fetch_k=fetch_k)
        for d in retrieved:
            key = ((d.page_content or "")[:128], d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                candidates.append(d)
            if len(candidates) >= k:
                break
    return candidates[:k]


def infer_doc_title(chunks: List[Document], fallback_name: str) -> str:
    for d in chunks[:3]:
        txt = (d.page_content or "").strip()
        if txt:
            first = txt.splitlines()[0].strip()
            return (first[:80] + "…") if len(first) > 80 else first
    return fallback_name
