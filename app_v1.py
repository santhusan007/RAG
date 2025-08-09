# Streamlit RAG App (Gemini version, single file)
#
# Dependencies (install these in your environment):
# - streamlit
# - pandas
# - pdfplumber
# - pypdf
# - langchain-core
# - langchain-text-splitters (comes via langchain)
# - langchain-chroma
# - langchain-huggingface
# - langchain-google-genai
# - chromadb
#
# Env var or Streamlit secret required:
# - GOOGLE_API_KEY (in environment or st.secrets["GOOGLE_API_KEY"]) 

import io
import os
import re
import hashlib
import tempfile
from typing import Any, List

import streamlit as st
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------- UI helpers ----------------

def set_page():
    st.set_page_config(page_title="Data ", page_icon="📄", layout="wide")
    st.markdown(
        """
        <style>
        .status { background: #eef8ff; color: #0b63ce; padding: 6px 10px; border-radius: 6px; font-size: 0.9rem; display: inline-block; }
        .app-banner {
            padding: 14px 16px; border-radius: 10px; margin-bottom: 12px;
            background: linear-gradient(90deg, #0b63ce, #1e90ff);
            color: white; font-weight: 600; display:flex; align-items:center; justify-content:space-between;
        }
        .app-banner .title { font-size: 1.05rem; letter-spacing: 0.3px; }
        .app-banner .tag { font-size: 0.85rem; opacity: 0.9; background: rgba(255,255,255,0.18); padding: 4px 8px; border-radius: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_banner():
    st.markdown(
        """
        <div class="app-banner">
          <div class="title">Atlas </div>
          <div class="tag">PDF + Spreadsheet </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- PDF & RAG utils (inlined) ----------------

def clean_text(text: str) -> str:
    if not text:
        return ""
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
        # Populate if empty
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
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=base_dir)
        return vs


def search_documents(vectorstore: Chroma, query: str, k: int = 6, method: str = "auto", fetch_k: int | None = None) -> List[Document]:
    try:
        if method == "mmr":
            fk = fetch_k if fetch_k is not None else max(k + 2, int(1.5 * k))
            return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fk)
        return vectorstore.similarity_search(query, k=k)
    except Exception:
        return vectorstore.similarity_search(query, k=k)


def smart_search(
    vectorstore: Chroma,
    query: str,
    k: int = 6,
    method: str = "auto",
    fetch_k: int | None = None,
    chunks: List[Document] | None = None,
) -> List[Document]:
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


# ---------------- Dataset utils (lightweight, inlined) ----------------

def df_profile_snippet(df: pd.DataFrame, max_cols: int = 12, max_rows: int = 6) -> str:
    parts = []
    parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols = list(df.columns)[:max_cols]
    parts.append("Columns: " + ", ".join(str(c) for c in cols) + (" …" if df.shape[1] > max_cols else ""))
    miss = df.isna().mean().sort_values(ascending=False)[:8]
    miss_txt = ", ".join([f"{c}: {v:.0%}" for c, v in miss.items()]) if not miss.empty else "(none)"
    parts.append("Missing (top): " + miss_txt)
    sample = df.head(max_rows).to_csv(index=False)
    parts.append("Sample (CSV):\n" + sample)
    return "\n".join(parts)


def analyze_spreadsheet(df: pd.DataFrame) -> str:
    try:
        desc = df.describe(include="all").transpose().head(8).fillna("")
        lines = ["Quick insights:"]
        lines.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        lines.append(f"- Numeric columns: {len(num_cols)}; Categorical columns: {len(cat_cols)}")
        if cat_cols:
            top = df[cat_cols[0]].astype(str).value_counts().head(5)
            lines.append(f"- Top values in {cat_cols[0]}: " + ", ".join([f"{k} ({v})" for k, v in top.items()]))
        return "\n".join(lines)
    except Exception:
        return "(Could not derive insights)"


def answer_dataset_question(df: pd.DataFrame, question: str) -> str | None:
    """Heuristic NL -> pandas answers using the full DataFrame.

    Handles common tasks:
    - row/column counts
    - mean/avg, sum/total, min, max of a numeric column
    - most frequent/mode of a categorical column
    - count/distribution by a categorical column (top 10)
    - top N <group> by <metric> (e.g., top 5 sub-category by sales)
    """
    try:
        q_raw = question or ""
        q = q_raw.lower()

        # 1) Basic shape
        if ("rows" in q and ("how many" in q or "count" in q)) or re.search(r"\brow count\b", q):
            return f"Rows: {df.shape[0]}"
        if ("columns" in q or "cols" in q) and ("how many" in q or "count" in q):
            return f"Columns: {df.shape[1]}"

        # Helpers for column matching
        def _norm(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

        col_map = {c: _norm(str(c)) for c in df.columns}

        def _find_col(name: str, prefer_numeric: bool | None = None) -> str | None:
            tokens = [t for t in _norm(name).split() if t]
            if not tokens:
                return None
            best, best_score = None, -1
            for c, cnorm in col_map.items():
                score = 0
                for t in tokens:
                    if t in cnorm:
                        score += 1
                # slight boost if last token matches ending
                if cnorm.endswith(tokens[-1]):
                    score += 0.5
                if prefer_numeric is True and pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                if prefer_numeric is False and not pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                if score > best_score:
                    best, best_score = c, score
            return best

        def _parse_n(text: str, default: int = 5) -> int:
            m = re.search(r"\btop\s*(\d{1,3})\b", text)
            if m:
                try:
                    return max(1, int(m.group(1)))
                except Exception:
                    pass
            return default

        # Helpers to infer likely metric/group columns by synonyms
        def _choose_metric(prefer: list[str] | None = None) -> str | None:
            prefer = prefer or ["sales", "revenue", "amount", "net sales", "qty", "quantity", "units"]
            # score columns: presence of synonym tokens + numeric dtype
            best, best_score = None, -1.0
            for c, cnorm in col_map.items():
                score = 0.0
                for i, key in enumerate(prefer[::-1]):
                    if key in cnorm:
                        score += 1.0 + i * 0.1
                if pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.5
                if score > best_score:
                    best, best_score = c, score
            return best

        def _choose_group(prefer: list[str] | None = None) -> str | None:
            prefer = prefer or ["product name", "product", "item", "sku", "sub-category", "subcategory", "category"]
            best, best_score = None, -1.0
            for c, cnorm in col_map.items():
                score = 0.0
                for i, key in enumerate(prefer[::-1]):
                    if key in cnorm:
                        score += 1.0 + i * 0.1
                if not pd.api.types.is_numeric_dtype(df[c]):
                    score += 0.25
                # penalize IDs unless name-like too
                if "id" in cnorm and not any(k in cnorm for k in ("name", "product")):
                    score -= 0.5
                if score > best_score:
                    best, best_score = c, score
            return best

        # 2) Top N <group> by <metric>
        m = re.search(r"top\s*(\d{1,3})\s+([a-z0-9 _\-/]+?)\s+by\s+([a-z0-9 _\-/]+)", q)
        if m:
            n = int(m.group(1))
            group_name = m.group(2).strip()
            metric_name = m.group(3).strip()
            group_col = _find_col(group_name, prefer_numeric=False)
            metric_col = _find_col(metric_name, prefer_numeric=True)
            if group_col and metric_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        # 2b) "Top selling products" / "best selling products" (no explicit metric/by clause)
        if ("top" in q or "best" in q) and ("sell" in q) and ("product" in q or "item" in q):
            n = _parse_n(q, 5)
            group_col = _choose_group(["product name", "product", "item", "sku"])
            metric_col = _choose_metric(["sales", "revenue", "amount", "qty", "quantity", "units"])
            if group_col and metric_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(n)
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        # 3) Most frequent / mode
        m = re.search(r"(most\s+(?:common|frequent)\s+|mode\s+of\s+)([a-z0-9 _\-/]+)", q)
        if m:
            col_name = m.group(2).strip()
            col = _find_col(col_name, prefer_numeric=False)
            if col:
                vc = df[col].astype(str).value_counts().head(_parse_n(q, 5))
                items = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                return f"Most frequent in {col}: {items}"

        # 4) Count by <column> / distribution of <column>
        m = re.search(r"(count\s+by|distribution\s+of)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col_name = m.group(2).strip()
            col = _find_col(col_name, prefer_numeric=False)
            if col:
                vc = df[col].astype(str).value_counts().head(_parse_n(q, 10))
                items = ", ".join([f"{k} ({v})" for k, v in vc.items()])
                return f"Counts by {col}: {items}"

        # 5) Aggregates: mean/avg, sum/total, min, max
        # mean/avg
        m = re.search(r"(average|avg|mean)\s+of\s+([a-z0-9 _\-/]+)", q)
        if not m:
            m = re.search(r"what\s+is\s+the\s+(average|avg|mean)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Average {col}: {df[col].dropna().mean():,.2f}"

        # sum/total
        m = re.search(r"(sum|total|overall)\s+of\s+([a-z0-9 _\-/]+)", q)
        if not m:
            m = re.search(r"what\s+is\s+the\s+(sum|total)\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Total {col}: {df[col].dropna().sum():,.2f}"

        # min
        m = re.search(r"(min|minimum|lowest)\s+of\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Minimum {col}: {df[col].dropna().min():,.2f}"

        # max
        m = re.search(r"(max|maximum|highest)\s+of\s+([a-z0-9 _\-/]+)", q)
        if m:
            col = _find_col(m.group(2).strip(), prefer_numeric=True)
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return f"Maximum {col}: {df[col].dropna().max():,.2f}"

    # Special: most profitable/most sales by <group>
        m = re.search(r"most\s+(profitable|revenue|sales)\s+(?:by|for)\s+([a-z0-9 _\-/]+)", q)
        if m:
            metric_hint = m.group(1)
            group_name = m.group(2).strip()
            metric_col = _find_col("profit" if "profit" in metric_hint else "sales", prefer_numeric=True)
            group_col = _find_col(group_name, prefer_numeric=False)
            if metric_col and group_col and pd.api.types.is_numeric_dtype(df[metric_col]):
                top = (
                    df[[group_col, metric_col]]
                    .dropna()
                    .groupby(group_col)[metric_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(_parse_n(q, 5))
                )
                items = ", ".join([f"{idx}: {val:,.2f}" for idx, val in top.items()])
                return f"Top {len(top)} {group_col} by {metric_col}: {items}"

        return None
    except Exception:
        return None


# ---------------- Gemini LLM helper ----------------

def get_gemini(model: str = "gemini-2.5-flash", temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    # Prefer environment variable to avoid raising when no secrets.toml exists
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            # st.secrets raises if no secrets file is configured
            api_key = st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing GOOGLE_API_KEY. Set it as an environment variable or in .streamlit/secrets.toml.")
        st.stop()
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


def llm_invoke(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    res = llm.invoke(prompt)
    try:
        # AIMessage
        content = getattr(res, "content", None)
        if isinstance(content, list):
            # If content is a list of parts, join text parts
            text = "".join([getattr(p, "text", "") if hasattr(p, "text") else str(p) for p in content])
            return text.strip()
        if isinstance(content, str) and content:
            return content.strip()
    except Exception:
        pass
    if isinstance(res, str):
        return res.strip()
    return str(res).strip()


# ---------------- Core app logic (Gemini) ----------------

def _file_ext(name: str) -> str:
    return (name or "").lower().split(".")[-1] if "." in (name or "") else ""


def ensure_vectorstore(file_name: str, file_bytes: bytes) -> None:
    file_id = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.get("file_id") != file_id or st.session_state.get("vectorstore") is None:
        st.session_state.messages = []
        chunks = process_pdf(file_bytes)
        if not chunks:
            st.error("Could not extract text from PDF.")
            st.stop()
        for c in chunks:
            meta = dict(c.metadata or {})
            meta["source"] = file_name
            c.metadata = meta
        st.session_state.vectorstore = get_or_build_vectorstore(chunks, file_id)
        st.session_state.doc_title = infer_doc_title(chunks, file_name)
        pages = {c.metadata.get("page") for c in chunks if isinstance(c.metadata, dict) and c.metadata.get("page")}
        st.session_state.page_count = len(pages) if pages else None
        st.session_state._raw_chunks = chunks
    # Reset any cached context to avoid stale summaries
    st.session_state.last_context_docs = []
    st.session_state.last_context_query = None
    st.session_state.last_context_file_id = file_id
    st.session_state.file_id = file_id


def summarize_document_full(vectorstore: Any, llm: ChatGoogleGenerativeAI, max_chunks: int = 30) -> str:
    seeds = [
        "overview and introduction",
        "key concepts and definitions",
        "main topics and sections",
        "methods and procedures",
        "results and findings",
        "conclusions and recommendations",
    ]
    seen = set()
    pool: List[Document] = []
    for q in seeds:
        for d in search_documents(vectorstore, q, k=8):
            key = hash((d.page_content[:256], d.metadata.get("page")))
            if key not in seen:
                seen.add(key)
                pool.append(d)
            if len(pool) >= max_chunks:
                break
        if len(pool) >= max_chunks:
            break
    if len(pool) < min(10, max_chunks):
        for d in vectorstore.similarity_search("document summary", k=max_chunks - len(pool)):
            key = hash((d.page_content[:256], d.metadata.get("page")))
            if key not in seen:
                seen.add(key)
                pool.append(d)
    context_parts, total = [], 0
    for d in pool:
        p = (d.page_content or "").strip()
        if not p:
            continue
        if total + len(p) > 6000:
            p = p[: max(0, 6000 - total)]
        context_parts.append(p)
        total += len(p)
        if total >= 6000:
            break
    context = "\n\n".join(context_parts)
    prompt = f"""
You are an expert technical writer. Create a clear, well-structured summary of the document below.

Requirements:
- Start with a one-paragraph executive overview.
- Then provide 4-8 concise bullet points covering key sections, definitions, and takeaways.
- If content is missing, omit it; do not hallucinate.

Document content:
{context}
"""
    return llm_invoke(llm, prompt)


# ---------- App ----------
set_page()
app_banner()

# Session state init
for key, default in (
    ("messages", []),
    ("file_id", None),
    ("vectorstore", None),
    ("doc_title", None),
    ("page_count", None),
    ("df", None),
    ("_uploaded_bytes", None),
    ("selected_category_col", "(auto)"),
    ("k_results", 5),
    ("show_sources", True),
    ("retrieval_method", "similarity"),
    ("last_context_docs", []),
    ("last_context_query", None),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar controls
with st.sidebar:
    st.header("Tools")
    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload a document (PDF/CSV/XLSX)",
        type=["pdf", "csv", "xlsx"],
        accept_multiple_files=False,
    )
    mode = st.radio("Mode", ["Chat", "Summarize"], index=0)
    st.selectbox(
        "Retrieval mode (PDF)", ["similarity", "mmr"],
        index=0 if st.session_state.get("retrieval_method") == "similarity" else 1,
        key="retrieval_method",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []

    # Spreadsheet helper: cache bytes & preview options
    selected_category_col = st.session_state.get("selected_category_col", "(auto)")
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        st.session_state["_uploaded_bytes"] = file_bytes
        ext = _file_ext(uploaded_file.name)
        if ext in ("csv", "xlsx"):
            try:
                if ext == "csv":
                    df_tmp = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    df_tmp = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
                st.session_state.df = df_tmp
                cat_cols = df_tmp.select_dtypes(include=["object", "category"]).columns.tolist()
                def _ok(c: str) -> bool:
                    s = df_tmp[c].astype(str)
                    nuniq = s.nunique(dropna=True)
                    if not (1 < nuniq <= 30):
                        return False
                    bad = ("name", "email", "id")
                    return not any(b in c.lower() for b in bad)
                options = ["(auto)"] + [c for c in cat_cols if _ok(c)]
                selected_category_col = st.selectbox("Category for bar chart", options, index=0, key="cat_select")
                st.session_state.selected_category_col = selected_category_col
            except Exception:
                pass

# Main
if uploaded_file is not None:
    ext = _file_ext(uploaded_file.name)
    file_bytes = st.session_state.get("_uploaded_bytes") or uploaded_file.getvalue()

    # Size limit for spreadsheets
    if ext in ("csv", "xlsx"):
        size = len(file_bytes)
        if size > 5 * 1024 * 1024:
            st.error("File too large. Please upload a CSV/XLSX up to 5 MB.")
            st.stop()

    llm = get_gemini(model="gemini-1.5-flash", temperature=0.1)

    # PDF branch
    if ext == "pdf":
        ensure_vectorstore(uploaded_file.name, file_bytes)
        vectorstore = st.session_state.vectorstore

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Document: {st.session_state.get('doc_title') or uploaded_file.name}")
            if st.session_state.get("page_count"):
                st.caption(f"Pages: {st.session_state.page_count}")
        with col2:
            st.markdown("<div class='status'>Ready • Indexed</div>", unsafe_allow_html=True)

        with st.container():
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                st.session_state.k_results = st.slider("Results to retrieve (k)", 2, 12, st.session_state.k_results, help="Top chunks to pass to the model (PDF only)")
            with rc2:
                st.session_state.show_sources = st.checkbox("Show source pages", value=st.session_state.show_sources)

        if mode == "Summarize":
            with st.status("Summarizing document…", expanded=True) as status:
                st.write("Preparing context…")
                try:
                    current_fid = st.session_state.get("file_id")
                    if st.session_state.get("last_context_docs") and st.session_state.get("last_context_file_id") == current_fid:
                        pool = st.session_state.last_context_docs
                    else:
                        seeds = ["overview", "introduction", "method", "results", "conclusion"]
                        seen = set()
                        pool = []
                        for q in seeds:
                            docs = smart_search(vectorstore, q, k=4, method=st.session_state.retrieval_method, chunks=st.session_state.get("_raw_chunks"))
                            for d in docs:
                                key = hash((d.page_content[:256], d.metadata.get("page")))
                                if key not in seen:
                                    seen.add(key)
                                    pool.append(d)
                            if len(pool) >= max(10, st.session_state.k_results):
                                break
                        st.session_state.last_context_docs = pool
                        st.session_state.last_context_file_id = current_fid

                    parts, total = [], 0
                    for d in pool:
                        p = (d.page_content or "").strip()
                        if not p:
                            continue
                        if total + len(p) > 6000:
                            p = p[: max(0, 6000 - total)]
                        parts.append(p)
                        total += len(p)
                        if total >= 6000:
                            break
                    context = "\n\n".join(parts)
                    prompt = f"""
You are an expert technical writer.
Write a concise summary of the document in 3–5 sentences covering the key sections and takeaways.
- Be specific and on-topic; avoid filler.
- Do not include bullet points.
- If content is missing, say so; do not hallucinate.

Document content:
{context}
"""
                    answer = llm_invoke(llm, prompt)
                except Exception as e:
                    answer = f"Error during summarization: {e}"
                status.update(label="Summary ready", state="complete")
            st.markdown(answer)
        else:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            user_question = st.chat_input("Ask anything about this document…")
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.spinner("Thinking..."):
                    try:
                        intent_summary = bool(re.search(r"\b(summarize|summary|in \d+\s*sentences)\b", user_question, re.IGNORECASE))
                        docs = None
                        current_fid = st.session_state.get("file_id")
                        if intent_summary and st.session_state.get("last_context_docs") and st.session_state.get("last_context_file_id") == current_fid:
                            docs = st.session_state.last_context_docs
                        else:
                            docs = smart_search(
                                vectorstore,
                                user_question,
                                k=st.session_state.k_results,
                                method=st.session_state.retrieval_method,
                                fetch_k=max(st.session_state.k_results + 2, int(1.5 * st.session_state.k_results)) if st.session_state.retrieval_method == "mmr" else None,
                                chunks=st.session_state.get("_raw_chunks"),
                            )
                        if docs:
                            parts, total = [], 0
                            for d in docs:
                                p = (d.page_content or "").strip()
                                if not p:
                                    continue
                                if total + len(p) > 4500:
                                    p = p[: max(0, 4500 - total)]
                                parts.append(p)
                                total += len(p)
                                if total >= 4500:
                                    break
                            context = "\n\n".join(parts)
                            st.session_state.last_context_docs = docs
                            st.session_state.last_context_query = user_question
                            st.session_state.last_context_file_id = current_fid

                            if intent_summary:
                                m = re.search(r"\bin\s*(\d+)\s*sentences\b", user_question, re.IGNORECASE)
                                n_sent = int(m.group(1)) if m else None
                                ask = f"Summarize the context{' in ' + str(n_sent) + ' sentences' if n_sent else ' in 3–5 sentences'}. Be concise and avoid hallucinations.".strip()
                                prompt = f"""
Summarize the following context clearly and concisely. If specific sentence count requested, adhere to it. Do not invent facts.

Context:
{context}

Task: {ask}
"""
                                answer = llm_invoke(llm, prompt)
                            else:
                                prompt = f"""
Use the provided context to answer the question. If not found, say you cannot find it in the document.

Context:
{context}

Question: {user_question}
"""
                                answer = llm_invoke(llm, prompt)
                            if st.session_state.show_sources:
                                pages = [str(d.metadata.get("page")) for d in docs if d.metadata.get("page")]
                                if pages:
                                    answer += f"\n\nSources: pages {', '.join(sorted(set(pages)))}"
                        else:
                            if intent_summary:
                                docs = smart_search(
                                    vectorstore,
                                    st.session_state.last_context_query or "document overview",
                                    k=st.session_state.k_results,
                                    method=st.session_state.retrieval_method,
                                    chunks=st.session_state.get("_raw_chunks"),
                                )
                                if docs:
                                    st.session_state.last_context_docs = docs
                                    parts = [(d.page_content or "").strip() for d in docs if (d.page_content or "").strip()]
                                    context = "\n\n".join(parts)[:4500]
                                    prompt = f"Summarize the context in 3 sentences.\n\nContext:\n{context}"
                                    answer = llm_invoke(llm, prompt)
                                else:
                                    answer = "I cannot find this information in the provided document."
                            else:
                                answer = "I cannot find this information in the provided document."
                    except Exception as e:
                        answer = f"Error: {e}"
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)

    # Spreadsheet branch
    elif ext in ("csv", "xlsx"):
        try:
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        except Exception as e:
            st.error(f"Failed to read spreadsheet: {e}")
            st.stop()
        st.session_state.df = df

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Spreadsheet: {uploaded_file.name}")
            st.caption(f"Rows: {df.shape[0]} • Columns: {df.shape[1]}")
        with col2:
            st.markdown("<div class='status'>Ready • Parsed</div>", unsafe_allow_html=True)

        insights = analyze_spreadsheet(df)
        st.markdown(insights)
        st.dataframe(df.head(15), use_container_width=True)

        # Visuals and helpers
        try:
            def _select_categorical_for_bar(df: pd.DataFrame) -> str | None:
                prefer = [
                    "role group", "role", "designation", "gender", "employment type",
                    "office location (actual)", "office location2", "office location", "office country",
                    "office city", "performance segment", "skill set", "age group", "cost center",
                ]
                bad = ["name", "email", "id"]
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                candidates = []
                for c in cat_cols:
                    s = df[c].astype(str)
                    nunique = s.nunique(dropna=True)
                    if 1 < nunique <= 30 and s.notna().mean() > 0.7 and not any(b in c.lower() for b in bad):
                        candidates.append(c)
                def score(c: str) -> int:
                    name = c.lower()
                    for i, kw in enumerate(prefer[::-1]):
                        if kw in name:
                            return i + 1
                    return 0
                if candidates:
                    candidates.sort(key=lambda x: (-score(x), df[x].astype(str).nunique()))
                    return candidates[0]
                return None

            def _select_datetime_column(df: pd.DataFrame) -> str | None:
                dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
                if dt_cols:
                    return dt_cols[0]
                for c in df.columns:
                    if any(k in c.lower() for k in ("date", "joining", "leaving", "dob", "month", "year", "running month")):
                        parsed = pd.to_datetime(df[c], errors="coerce")
                        if parsed.notna().any():
                            df[c + "_parsed_dt"] = parsed
                            return c + "_parsed_dt"
                return None

            def _select_numeric_for_timeseries(df: pd.DataFrame) -> str | None:
                pref = ["no of days", "days", "experience", "time with us", "time", "age", "count", "number"]
                avoid = ["id"]
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                scored = []
                for c in num_cols:
                    name = c.lower()
                    if any(b in name for b in avoid):
                        continue
                    s = df[c]
                    if s.dropna().nunique() <= 1:
                        continue
                    kw_score = 0
                    for i, kw in enumerate(pref[::-1]):
                        if kw in name:
                            kw_score = i + 1
                            break
                    scored.append((kw_score, -s.dropna().nunique(), c))
                if scored:
                    scored.sort(reverse=True)
                    return scored[0][2]
                for c in num_cols:
                    if "id" not in c.lower():
                        return c
                return num_cols[0] if num_cols else None

            # Inline category selector
            selected = st.session_state.get("selected_category_col")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            def _ok(c: str) -> bool:
                s = df[c].astype(str)
                nuniq = s.nunique(dropna=True)
                if not (1 < nuniq <= 30):
                    return False
                bad = ("name", "email", "id")
                return not any(b in c.lower() for b in bad)
            options_inline = ["(auto)"] + [c for c in cat_cols if _ok(c)]
            default_idx = options_inline.index(selected) if selected in options_inline else 0
            chosen = st.selectbox("Category for bar chart", options_inline, index=default_idx, key="cat_select_inline")
            if chosen != "(auto)":
                st.session_state.selected_category_col = chosen
                selected = chosen

            cat_col = selected if selected and selected != "(auto)" and selected in df.columns else _select_categorical_for_bar(df)
            if cat_col:
                vc = df[cat_col].astype(str).value_counts().head(10)
                st.subheader(f"Top categories in {cat_col}")
                st.bar_chart(vc)

            dcol = _select_datetime_column(df)
            ncol = _select_numeric_for_timeseries(df)
            if dcol and ncol:
                ts = (
                    df[[dcol, ncol]].dropna()
                    .assign(__period=lambda x: x[dcol].dt.to_period('M').dt.to_timestamp())
                    .groupby("__period")[ncol]
                    .mean()
                )
                if not ts.empty:
                    st.subheader(f"Time series: mean {ncol} by month")
                    st.line_chart(ts)

            # Quick breakdowns (best-effort)
            try:
                if "Role" in df.columns or "Role Group" in df.columns:
                    group_col = "Role" if "Role" in df.columns else "Role Group"
                    st.subheader(f"Headcount by {group_col}")
                    st.bar_chart(df[group_col].astype(str).value_counts().head(15))
                if "Gender" in df.columns:
                    st.subheader("Headcount by Gender")
                    st.bar_chart(df["Gender"].astype(str).value_counts())
            except Exception:
                pass
        except Exception:
            pass

        if mode == "Summarize":
            with st.status("Summarizing dataset…", expanded=False) as status:
                try:
                    context = df_profile_snippet(df, max_cols=12, max_rows=6)
                    dims = []
                    for cand in ["Role", "Role Group", "Designation", "Gender", "Employment Type", "Office Location (Actual)", "Office Country", "Office city", "Performance Segment", "Age Group"]:
                        if cand in df.columns:
                            dims.append(cand)
                    dim_text = ("Focus on dimensions: " + ", ".join(dims) + ".") if dims else ""
                    prompt = f"""
You are a data analyst. Summarize the dataset described below. Include key columns, notable distributions, missing data, and 3-6 high-level insights. {dim_text} Avoid hallucinations.

Dataset profile:
{context}
"""
                    answer = llm_invoke(llm, prompt)
                except Exception as e:
                    answer = f"Error during summarization: {e}"
                status.update(label="Summary ready", state="complete")
            st.markdown(answer)
        else:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            user_question = st.chat_input("Ask about this dataset…")
            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)
                with st.spinner("Thinking..."):
                    try:
                        structured = answer_dataset_question(df, user_question)
                        if structured is not None:
                            answer = structured
                        else:
                            context = df_profile_snippet(df, max_cols=12, max_rows=6)
                            prompt = f"""
Use the dataset profile and sample rows to answer the user's question. If the answer isn't apparent from the provided data, say so.

Dataset profile:
{context}

Question: {user_question}
"""
                            answer = llm_invoke(llm, prompt)
                    except Exception as e:
                        answer = f"Error: {e}"
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
    else:
        st.error("Unsupported file type. Please upload a PDF, CSV, or XLSX file.")
else:
    # Memory-only chat when no document is uploaded
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    user_question = st.chat_input("Ask anything… (no document loaded)")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.spinner("Thinking..."):
            try:
                hist_msgs = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")][-6:]
                history = []
                for i in range(0, len(hist_msgs), 2):
                    try:
                        qh = hist_msgs[i]["content"]
                        ah = hist_msgs[i+1]["content"]
                        history.append(f"User: {qh}\nAssistant: {ah}")
                    except Exception:
                        pass
                history_text = "\n".join(history)
                prompt = f"""
You are a helpful assistant. There is no document context loaded. Answer the user's question using only the conversation so far.

Conversation so far (may be empty):
{history_text}

Question: {user_question}

Answer concisely:
"""
                llm = get_gemini(model="gemini-1.5-flash", temperature=0.2)
                answer = llm_invoke(llm, prompt)
            except Exception as e:
                answer = f"Error: {e}"
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
