import os
import re
import tempfile
import difflib
from typing import List, Tuple, Optional

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

    # Optional OCR fallback for scanned PDFs if we still have very little text
    try:
        total_len = sum(len(d.page_content or "") for d in docs)
        if total_len < 50:
            try:
                from pdf2image import convert_from_path  # type: ignore
                import pytesseract  # type: ignore
                ocr_pages: List[Document] = []
                images = convert_from_path(tmp_path)
                for i, img in enumerate(images, start=1):
                    try:
                        txt = pytesseract.image_to_string(img) or ""
                        txt = clean_text(txt)
                        if txt:
                            ocr_pages.append(Document(page_content=txt, metadata={"page": i}))
                    except Exception:
                        continue
                if ocr_pages:
                    docs = ocr_pages
            except Exception:
                # OCR tools not installed; skip quietly
                pass
    except Exception:
        pass

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

    # Try explicit PersistentClient to avoid tenant/database issues on newer Chroma
    client = None
    try:
        import chromadb  # type: ignore
        try:
            from chromadb.config import Settings  # type: ignore
            settings = None
            try:
                settings = Settings(anonymized_telemetry=False, allow_reset=True, persist_directory=base_dir)
            except Exception:
                settings = None
            try:
                if settings is not None:
                    client = chromadb.PersistentClient(settings=settings)
                else:
                    client = chromadb.PersistentClient(path=base_dir)
            except Exception:
                try:
                    client = chromadb.PersistentClient(path=base_dir)
                except Exception:
                    client = None
            if client is not None:
                try:
                    if hasattr(client, "get_tenant") and hasattr(client, "create_tenant"):
                        try:
                            client.get_tenant("default_tenant")
                        except Exception:
                            client.create_tenant("default_tenant")
                    if hasattr(client, "get_database") and hasattr(client, "create_database"):
                        try:
                            client.get_database("default_database")
                        except Exception:
                            client.create_database("default_database")
                except Exception:
                    pass
        except Exception:
            client = None
    except Exception:
        client = None

    if client is not None:
        try:
            vs = Chroma(client=client, collection_name=f"rag_{file_id}", embedding_function=embeddings)
            try:
                count = 0
                try:
                    count = vs._collection.count()  # type: ignore[attr-defined]
                except Exception:
                    count = 0
                if count == 0:
                    vs.add_documents(chunks)
            except Exception:
                pass
            return vs
        except Exception:
            pass

    # Fallback to legacy persistent path
    try:
        vs = Chroma(persist_directory=base_dir, embedding_function=embeddings)
        try:
            count = 0
            try:
                count = vs._collection.count()  # type: ignore[attr-defined]
            except Exception:
                count = 0
            if count == 0:
                vs.add_documents(chunks)
        except Exception:
            pass
        return vs
    except Exception:
        # Final fallback: in-memory (non-persistent)
        return Chroma.from_documents(chunks, embeddings)


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
            # exact or fuzzy match boost
            match = phrase in text
            if not match:
                try:
                    ratio = difflib.SequenceMatcher(None, phrase, text[: min(4000, len(text))]).quick_ratio()
                    match = ratio >= 0.65
                except Exception:
                    match = False
            if match:
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


# --------- Section-aware retrieval helpers ---------

def _group_text_by_page(chunks: List[Document]) -> List[Tuple[int, str]]:
    pages: dict[int, List[str]] = {}
    for d in chunks:
        p = int(d.metadata.get("page") or 0)
        pages.setdefault(p, []).append(d.page_content or "")
    grouped = []
    for p in sorted(pages.keys()):
        text = "\n".join(pages[p])
        grouped.append((p, text))
    return grouped


def _is_header_line(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    if len(line) > 140:
        return False
    # header heuristics: title-case or uppercase, numbered, or ends with colon
    if line.endswith(":") and len(line) >= 5:
        return True
    if re.match(r"^(\d+\.|[ivx]+\.|[a-z]\))\s+", line.lower()):
        return True
    letters = re.sub(r"[^A-Za-z]", "", line)
    if letters and letters.isupper() and len(letters) >= 4:
        return True
    # Title Case ratio
    words = line.split()
    if words and sum(1 for w in words if w[:1].isupper()) / max(1, len(words)) >= 0.7:
        return True
    return False


def _find_best_header(lines: List[str], query: str) -> Optional[int]:
    q = re.sub(r"\s+", " ", query.strip())
    best_idx, best_score = None, 0.0
    for i, ln in enumerate(lines):
        if not _is_header_line(ln):
            continue
        try:
            score = difflib.SequenceMatcher(None, q.lower(), ln.lower()).ratio()
        except Exception:
            score = 0.0
        # small bonus if query words subset of header
        if all(w in ln.lower() for w in q.lower().split()[:2]):
            score += 0.05
        if score > best_score:
            best_idx, best_score = i, score
    if best_idx is not None and best_score >= 0.6:
        return best_idx
    return None


def _extract_section_span(pages: List[Tuple[int, str]], query: str, max_chars: int = 8000) -> Optional[Tuple[str, int, int, str]]:
    # Flatten to lines with (page, line_idx)
    lines_meta: List[Tuple[int, str]] = []
    for p, text in pages:
        for ln in (text or "").splitlines():
            lines_meta.append((p, ln))
    lines = [ln for _, ln in lines_meta]
    idx = _find_best_header(lines, query)
    if idx is None:
        return None
    start_page = lines_meta[idx][0]
    header_text = lines[idx].strip()
    # Collect until next header
    buf: List[str] = []
    total = 0
    for j in range(idx + 1, len(lines)):
        ln_page, ln_txt = lines_meta[j]
        if _is_header_line(ln_txt) and lines_meta[j][0] >= start_page:
            break
        piece = (ln_txt or "").rstrip()
        if not piece:
            continue
        if total + len(piece) > max_chars:
            break
        buf.append(piece)
        total += len(piece) + 1
    content = "\n".join(buf).strip()
    if not content:
        return None
    # Estimate end page as last page encountered
    end_page = start_page
    for j in range(idx + 1, min(idx + 1 + len(buf), len(lines_meta))):
        end_page = max(end_page, lines_meta[j][0])
    return content, start_page, end_page, header_text


def robust_retrieve(vectorstore: Chroma, query: str, chunks: List[Document] | None, k: int = 6, method: str = "auto") -> List[Document]:
    """Section-aware retrieval with fuzzy header detection + embedding fallback.

    - Fuzzy-match the query to likely header lines, then extract text until the next header across pages.
    - If not found, fall back to smart_search (keyword boost + embeddings).
    """
    results: List[Document] = []
    try:
        if chunks:
            pages = _group_text_by_page(chunks)
            span = _extract_section_span(pages, query)
            if span:
                content, p_start, p_end, header = span
                meta = {"page_start": p_start, "page_end": p_end, "section": header}
                results.append(Document(page_content=content, metadata=meta))
                # Also fetch a couple of embedding-nearby chunks to enrich context
                emb_docs = search_documents(vectorstore, query, k=max(2, k // 2), method=method)
                seen = set()
                for d in emb_docs:
                    key = hash((d.page_content[:256], d.metadata.get("page")))
                    if key not in seen:
                        seen.add(key)
                        results.append(d)
                return results[:k]
    except Exception:
        pass
    # Fallback: normal smart search
    return smart_search(vectorstore, query, k=k, method=method, chunks=chunks)


def infer_doc_title(chunks: List[Document], fallback_name: str) -> str:
    for d in chunks[:3]:
        txt = (d.page_content or "").strip()
        if txt:
            first = txt.splitlines()[0].strip()
            return (first[:80] + "…") if len(first) > 80 else first
    return fallback_name
