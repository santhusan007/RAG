import io
import hashlib
import re
import streamlit as st
import pandas as pd
from typing import Any
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from typing import List

from modules.pdf_utils import process_pdf, get_or_build_vectorstore, infer_doc_title, search_documents, smart_search, robust_retrieve
from modules.dataset_utils import analyze_spreadsheet, df_profile_snippet, answer_dataset_question
from modules.ui import set_page, header as ui_header, app_banner


# ---------- Small helpers ----------
def _file_ext(name: str) -> str:
    return (name or "").lower().split(".")[-1] if "." in (name or "") else ""


def ensure_vectorstore(file_name: str, file_bytes: bytes) -> bool:
    """Builds/loads vectorstore for a PDF and returns True if this is a new file.

    Also clears chat and cached context only when a new file is detected.
    """
    file_id = hashlib.md5(file_bytes).hexdigest()
    is_new = st.session_state.get("file_id") != file_id or st.session_state.get("vectorstore") is None
    if is_new:
        # Clear chat and cached context for new document
        st.session_state.messages = []
        st.session_state.last_context_docs = []
        st.session_state.last_context_query = None
        st.session_state.last_context_file_id = None

        # Build chunks/vector store
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
        # Tie caches to this file
        st.session_state.last_context_file_id = file_id
        st.session_state.file_id = file_id
        return True
    else:
        # Same file; keep existing vectorstore/chunks and just set active file id
        st.session_state.file_id = file_id
        return False


def summarize_document_full(vectorstore: Any, llm: Ollama, max_chunks: int = 30) -> str:
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
    return llm.invoke(prompt).strip()


# ---------- UI setup ----------
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
    ("last_context_docs", []),  # cache last retrieved docs (Documents)
    ("last_context_query", None),
    ("last_context_file_id", None),
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
    st.selectbox("Retrieval mode (PDF)", ["similarity", "mmr"], index=0 if st.session_state.get("retrieval_method")=="similarity" else 1, key="retrieval_method")
    if st.button("Clear chat"):
        st.session_state.messages = []

    # If spreadsheet, allow category column selection
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
                # Build candidate list
                cat_cols = df_tmp.select_dtypes(include=["object", "category"]).columns.tolist()
                # filter columns with reasonable cardinality and avoid id/name/email
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


# ---------- Main ----------
if uploaded_file is not None:
    ext = _file_ext(uploaded_file.name)
    file_bytes = st.session_state.get("_uploaded_bytes") or uploaded_file.getvalue()

    # Spreadsheet size limit
    if ext in ("csv", "xlsx"):
        size = len(file_bytes)
        if size > 5 * 1024 * 1024:
            st.error("File too large. Please upload a CSV/XLSX up to 5 MB.")
            st.stop()

    llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.1)

    # PDF branch
    if ext == "pdf":
        is_new = ensure_vectorstore(uploaded_file.name, file_bytes)
        vectorstore = st.session_state.vectorstore
        if is_new:
            st.info("Previous chat cleared — ready for new document analysis.")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Document: {st.session_state.get('doc_title') or uploaded_file.name}")
            if st.session_state.get("page_count"):
                st.caption(f"Pages: {st.session_state.page_count}")
        with col2:
            st.markdown("<div class='status'>Ready • Indexed</div>", unsafe_allow_html=True)

        # Inline retrieval controls (always visible under the header)
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
                    # Reuse last context only if it belongs to this file; otherwise derive fresh context
                    current_fid = st.session_state.get("file_id")
                    if st.session_state.get("last_context_docs") and st.session_state.get("last_context_file_id") == current_fid:
                        pool = st.session_state.last_context_docs
                    else:
                        # seed summary pool via fast similarity for speed
                        seeds = [
                            "overview",
                            "introduction",
                            "method",
                            "results",
                            "conclusion",
                        ]
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

                    # build context string
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
                    answer = llm.invoke(prompt).strip()
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
                        # Detect summarization intent and reuse last context if available for this file
                        intent_summary = bool(re.search(r"\b(summarize|summary|in \d+\s*sentences)\b", user_question, re.IGNORECASE))
                        docs = None
                        current_fid = st.session_state.get("file_id")
                        if intent_summary and st.session_state.get("last_context_docs") and st.session_state.get("last_context_file_id") == current_fid:
                            docs = st.session_state.last_context_docs
                        else:
                            docs = robust_retrieve(
                                vectorstore,
                                user_question,
                                k=st.session_state.k_results,
                                method=st.session_state.retrieval_method,
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
                            # cache context for follow-ups (e.g., summarization) tied to this file
                            st.session_state.last_context_docs = docs
                            st.session_state.last_context_query = user_question
                            st.session_state.last_context_file_id = current_fid

                            if intent_summary:
                                # Build a concise summary (default 3–5 sentences when not specified)
                                m = re.search(r"\bin\s*(\d+)\s*sentences\b", user_question, re.IGNORECASE)
                                n_sent = int(m.group(1)) if m else None
                                ask = f"Summarize the context{' in ' + str(n_sent) + ' sentences' if n_sent else ' in 3–5 sentences'}. Be concise and avoid hallucinations.".strip()
                                prompt = f"""
Summarize the following context clearly and concisely. If specific sentence count requested, adhere to it. Do not invent facts.

Context:
{context}

Task: {ask}
"""
                                answer = llm.invoke(prompt).strip()
                            else:
                                prompt = f"""
Use the provided context to answer the question. If not found, say you cannot find it in the document.

Context:
{context}

Question: {user_question}
"""
                                answer = llm.invoke(prompt).strip()
                            if st.session_state.show_sources:
                                labels = []
                                for d in docs:
                                    meta = d.metadata or {}
                                    if meta.get("page"):
                                        labels.append(str(meta.get("page")))
                                    elif meta.get("page_start") and meta.get("page_end"):
                                        labels.append(f"{meta['page_start']}–{meta['page_end']}")
                                if labels:
                                    answer += f"\n\nSources: pages {', '.join(sorted(set(labels)))}"
                        else:
                            # If we tried to reuse but found nothing, re-fetch normally once
                            if intent_summary:
                                docs = robust_retrieve(
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
                                    answer = llm.invoke(prompt).strip()
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

        # Reset chat/context if this is a new uploaded file (by content hash)
        file_id = hashlib.md5(file_bytes).hexdigest()
        if st.session_state.get("file_id") != file_id:
            st.session_state.messages = []
            st.session_state.last_context_docs = []
            st.session_state.last_context_query = None
            st.session_state.last_context_file_id = None
            st.session_state.file_id = file_id
            st.info("Previous chat cleared — ready for new document analysis.")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Spreadsheet: {uploaded_file.name}")
            st.caption(f"Rows: {df.shape[0]} • Columns: {df.shape[1]}")
        with col2:
            st.markdown("<div class='status'>Ready • Parsed</div>", unsafe_allow_html=True)

        insights = analyze_spreadsheet(df)
        st.markdown(insights)
        st.dataframe(df.head(15), use_container_width=True)

        # Visuals
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

            # Use user-selected category if available and show inline selector for visibility
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
                    answer = llm.invoke(prompt).strip()
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
Use the dataset profile and context to answer the user's question. If the answer isn't apparent from the provided data, say so without guessing.

Dataset profile:
{context}

Question: {user_question}
"""
                            answer = llm.invoke(prompt).strip()
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
                llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.2)
                answer = llm.invoke(prompt).strip()
            except Exception as e:
                answer = f"Error: {e}"
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
