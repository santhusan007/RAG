# Streamlit RAG App v2 (Gemini + OCR Field Extraction)
#
# New in v2:
# - "Extract Fields" mode: accepts PDF (text or scanned) and images (JPG/PNG), performs OCR when needed,
#   extracts key fields (IDs/numbers/dates/phones) and header-value pairs, and exports to CSV/Excel.
# - Reuses existing RAG and dataset features.
#
# Optional OCR dependencies:
#   pip install pytesseract pdf2image easyocr pillow
# Also install Tesseract (and Poppler for pdf2image) on your OS.
#
# Env:
#   GOOGLE_API_KEY must be set (or in .streamlit/secrets.toml)

import io
import os
import re
import hashlib
from typing import Any, List, Tuple, Optional, Dict

import streamlit as st
import pandas as pd
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.ui import set_page, header as ui_header, app_banner

# Reuse shared utilities
from modules.pdf_utils import (
    process_pdf,
    get_or_build_vectorstore,
    infer_doc_title,
    search_documents,
    robust_retrieve,
)
from modules.dataset_utils import (
    df_profile_snippet,
    analyze_spreadsheet,
    answer_dataset_question,
)


# (Using shared UI: set_page(), app_banner() from modules.ui)


# --------------- LLM helpers ---------------

def get_gemini(model: str = "gemini-2.5-flash", temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]  # type: ignore[index]
        except Exception:
            api_key = None
    if not api_key:
        st.error("Missing GOOGLE_API_KEY. Set it in env or .streamlit/secrets.toml.")
        st.stop()
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


def llm_invoke(llm: ChatGoogleGenerativeAI, prompt: str) -> str:
    res = llm.invoke(prompt)
    try:
        content = getattr(res, "content", None)
        if isinstance(content, list):
            txt = "".join([getattr(p, "text", "") if hasattr(p, "text") else str(p) for p in content])
            return txt.strip()
        if isinstance(content, str) and content:
            return content.strip()
    except Exception:
        pass
    if isinstance(res, str):
        return res.strip()
    return str(res).strip()


# --------------- Text utils ---------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def chunks_to_full_text(chunks: List[Document]) -> str:
    if not chunks:
        return ""
    by_page: Dict[int, List[str]] = {}
    for d in chunks:
        p = int(d.metadata.get("page") or 0)
        by_page.setdefault(p, []).append(d.page_content or "")
    parts: List[str] = []
    for p in sorted(by_page.keys()):
        parts.append("\n".join(by_page[p]))
    if not parts:
        parts = [c.page_content or "" for c in chunks]
    return "\n".join(parts)


# --------------- Image OCR ---------------

def ocr_image_bytes(file_bytes: bytes) -> str:
    """OCR an image with preprocessing. Prefers pytesseract; falls back to EasyOCR."""
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        return ""

    def preprocess_image_for_ocr(im: Image.Image) -> List[Image.Image]:
        """Generate a set of preprocessed variants to improve OCR on noisy scans."""
        variants: List[Image.Image] = []
        # Base grayscale
        gray = ImageOps.grayscale(im)
        variants.append(gray)
        # Scaled up
        try:
            scale = 1.5
            up = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.LANCZOS)
            variants.append(up)
        except Exception:
            pass
        # Contrast boost
        try:
            contrast = ImageEnhance.Contrast(gray).enhance(1.8)
            variants.append(contrast)
        except Exception:
            pass
        # Sharpen
        try:
            sharp = gray.filter(ImageFilter.SHARPEN)
            variants.append(sharp)
        except Exception:
            pass
        # Light denoise via median
        try:
            med = gray.filter(ImageFilter.MedianFilter(size=3))
            variants.append(med)
        except Exception:
            pass
        # Binary threshold (simple)
        try:
            bw = gray.point(lambda p: 255 if p > 160 else 0, mode="1").convert("L")
            variants.append(bw)
        except Exception:
            pass
        # Also try contrast+sharpen combo
        try:
            combo = ImageEnhance.Contrast(gray).enhance(2.0).filter(ImageFilter.SHARPEN)
            variants.append(combo)
        except Exception:
            pass
        return variants

    # Try pytesseract with multiple preprocess variants
    try:
        import pytesseract  # type: ignore
        for trial in preprocess_image_for_ocr(img):
            txt = clean_text(pytesseract.image_to_string(trial) or "")
            if len(txt) >= 8:
                return txt
        # Last try: original image if variants failed
        txt = clean_text(pytesseract.image_to_string(img) or "")
        if len(txt) >= 3:
            return txt
    except Exception:
        pass

    # Fallback EasyOCR
    try:
        import easyocr  # type: ignore
        reader = easyocr.Reader(["en"], gpu=False)
        result = reader.readtext(io.BytesIO(file_bytes), detail=0)
        return clean_text(" ".join(result))
    except Exception:
        return ""


# --------------- Field extraction ---------------

PAN_REGEX = re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b")
DATE_REGEX = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
PHONE_REGEX = re.compile(r"\b(\+?\d[\d\s\-]{8,}\d)\b")
NUM_LONG_REGEX = re.compile(r"\b(\d{6,})\b")

# Flexible label patterns (generic, extensible)
NAME_LABELS = [
    r"name of the holder", r"card ?holder name", r"cardholder name", r"full name",
    r"applicant name", r"customer name", r"name as.*?",
    r"name",  # keep generic last so specific ones match first
]
NAME_LINE_REGEX = re.compile(rf"(?i)\b({'|'.join(NAME_LABELS)})\b[\s:.-]*([A-Z][A-Za-z][A-Za-z .']{{1,}})")
FATHER_LABELS = [r"father'?s name", r"father name", r"parent'?s name"]
FATHER_LINE_REGEX = re.compile(rf"(?i)\b({'|'.join(FATHER_LABELS)})\b[\s:.-]*([A-Z][A-Za-z .']{{2,}})")


def split_lines(text: str) -> List[str]:
    raw = re.split(r"[\r\n]+", text or "")
    lines = [clean_text(x) for x in raw if clean_text(x)]
    return lines


def is_labelish(s: str) -> bool:
    if not s or len(s) > 60:
        return False
    if sum(ch.isdigit() for ch in s) > max(2, len(s) // 3):
        return False
    if ":" in s:
        return True
    words = s.split()
    if not words:
        return False
    if s.isupper() and len(words) <= 6:
        return True
    titled = sum(1 for w in words if w[:1].isupper())
    return titled / max(1, len(words)) >= 0.7 and len(words) <= 6


def extract_field_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    # colon-delimited
    for ln in lines:
        if ":" in ln:
            left, right = ln.split(":", 1)
            l, r = clean_text(left), clean_text(right)
            if l and r:
                pairs.append((l, r))
    # multi-space split on same line (label    value)
    for ln in lines:
        if ":" not in ln and re.search(r"\s{3,}", ln):
            parts = re.split(r"\s{3,}", ln.strip(), maxsplit=1)
            if len(parts) == 2:
                l, r = clean_text(parts[0]), clean_text(parts[1])
                if l and r:
                    pairs.append((l, r))
    # adjacency: labelish -> value
    for i, ln in enumerate(lines[:-1]):
        if is_labelish(ln) and ":" not in ln:
            nxt = lines[i + 1]
            if not is_labelish(nxt) and nxt != ln:
                pairs.append((ln, nxt))
    # dedupe
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for l, v in pairs:
        key = (l.lower(), v.lower())
        if key not in seen:
            seen.add(key)
            uniq.append((l, v))
    return uniq


def extract_entities(text: str) -> Dict[str, List[str]]:
    vals: Dict[str, List[str]] = {"PAN": [], "Dates": [], "Phones": [], "LongNumbers": []}
    vals["PAN"] = list(dict.fromkeys(PAN_REGEX.findall(text)))
    vals["Dates"] = list(dict.fromkeys(DATE_REGEX.findall(text)))
    vals["Phones"] = list(dict.fromkeys(PHONE_REGEX.findall(text)))
    vals["LongNumbers"] = list(dict.fromkeys(NUM_LONG_REGEX.findall(text)))
    return vals


def derive_key_fields(lines: List[str], pairs: List[Tuple[str, str]], entities: Dict[str, List[str]]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    # PAN
    if entities.get("PAN"):
        kv["PAN Number"] = entities["PAN"][0]
    # Name from labeled pairs first
    name_keys = [
        "name of the holder", "card holder name", "cardholder name", "full name",
        "applicant name", "customer name", "name as", "name",
    ]
    for l, v in pairs:
        low = l.lower()
        if any(k in low for k in name_keys):
            # avoid generic phrases
            val = v.strip().strip(':').strip()
            if len(val) >= 2 and not re.search(r"income tax|govt|government|permanent account|signature", val, re.I):
                kv["Name"] = val
                break
    # If still missing, scan lines with regex
    if "Name" not in kv:
        for ln in lines:
            m = NAME_LINE_REGEX.search(ln)
            if m:
                candidate = m.group(2).strip()
                if 2 <= len(candidate) <= 60 and not re.search(r"income tax|govt|government|permanent account|signature", candidate, re.I):
                    kv["Name"] = candidate
                    break
    # Heuristic near PAN: take nearby line that looks like a person name
    if "Name" not in kv and entities.get("PAN"):
        pan = entities["PAN"][0]
        idxs = [i for i, ln in enumerate(lines) if pan in ln]
        banned = re.compile(r"income tax|govt|government|permanent account|signature|date|address|father|mother|son of|daughter of", re.I)
        def looks_like_name(s: str) -> bool:
            s = s.strip(' :.-')
            if not (2 <= len(s) <= 60):
                return False
            if any(ch.isdigit() for ch in s):
                return False
            words = s.split()
            if len(words) > 7 or len(words) < 1:
                return False
            # uppercase heavy names are common on ID cards
            if banned.search(s):
                return False
            # require at least two alphabetic characters and mostly letters/spaces
            letters = sum(ch.isalpha() for ch in s)
            return letters >= max(2, int(0.6 * len(s)))
        for idx in idxs:
            for j in range(max(0, idx - 3), min(len(lines), idx + 4)):
                if j == idx:
                    continue
                cand = lines[j]
                if looks_like_name(cand):
                    kv["Name"] = cand.strip()
                    break
            if "Name" in kv:
                break
    # DOB
    dob_keys = ["dob", "date of birth", "birth"]
    for l, v in pairs:
        if any(k in l.lower() for k in dob_keys):
            kv["Date of Birth"] = v.strip()
            break
    if "Date of Birth" not in kv and entities.get("Dates"):
        kv["Date of Birth"] = entities["Dates"][0]
    # Father Name (if present)
    for l, v in pairs:
        if any(k in l.lower() for k in ["father", "father's name", "father name"]):
            kv["Father's Name"] = v.strip()
            break
    if "Father's Name" not in kv:
        for ln in lines:
            m = FATHER_LINE_REGEX.search(ln)
            if m:
                kv["Father's Name"] = m.group(2).strip()
                break
    # Phones and generic numbers (joined)
    if entities.get("Phones"):
        kv["Phone(s)"] = ", ".join(entities["Phones"][:3])
    # Generic long numbers that are not PAN
    longs = [n for n in (entities.get("LongNumbers") or []) if n not in (entities.get("PAN") or [])]
    if longs:
        kv["ID Number(s)"] = ", ".join(longs[:5])
    return kv


def extract_from_text(text: str) -> Dict[str, str]:
    lines = split_lines(text)
    pairs = extract_field_pairs(lines)
    ents = extract_entities(" ".join(lines))
    kv = derive_key_fields(lines, pairs, ents)
    # Also include other labeled pairs not already in kv (generic flexibility)
    for l, v in pairs:
        key = l.strip().rstrip(':').strip()
        if key and v and key not in kv:
            kv[key] = v
    return kv


# --------------- App logic ---------------

def _file_ext(name: str) -> str:
    return (name or "").lower().split(".")[-1] if "." in (name or "") else ""


def ensure_pdf_vectorstore(file_name: str, file_bytes: bytes) -> bool:
    """Build/load vectorstore for PDFs; return True if new file. Clears chat/context on new."""
    file_id = hashlib.md5(file_bytes).hexdigest()
    is_new = st.session_state.get("file_id") != file_id or st.session_state.get("vectorstore") is None
    if is_new:
        st.session_state.messages = []
        st.session_state.last_context_docs = []
        st.session_state.last_context_query = None
        st.session_state.last_context_file_id = None

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
        st.session_state.last_context_file_id = file_id
        st.session_state.file_id = file_id
        return True
    st.session_state.file_id = file_id
    return False


# --------------- Main ---------------

set_page()
app_banner()

# Session defaults
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
    ("last_context_file_id", None),
    ("doc_processing", False),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
with st.sidebar:
    st.header("Tools")
    uploaded = st.file_uploader(
        "Upload a document (PDF/CSV/XLSX/JPG/PNG)",
        type=["pdf", "csv", "xlsx", "jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    mode = st.radio("Mode", ["Chat", "Summarize", "Extract Fields"], index=0)
    st.selectbox(
        "Retrieval mode (PDF)", ["similarity", "mmr"],
        index=0 if st.session_state.get("retrieval_method") == "similarity" else 1,
        key="retrieval_method",
    )
    # PDF options moved to sidebar for compact control
    st.session_state.k_results = st.slider("Results to retrieve (k)", 2, 12, st.session_state.k_results)
    st.session_state.show_sources = st.checkbox("Show source pages", value=st.session_state.show_sources)
    if st.button("Clear chat"):
        st.session_state.messages = []

    # Spreadsheet helper
    selected_category_col = st.session_state.get("selected_category_col", "(auto)")
    if uploaded is not None:
        file_bytes_tmp = uploaded.getvalue()
        st.session_state["_uploaded_bytes"] = file_bytes_tmp
        ext_tmp = _file_ext(uploaded.name)
        if ext_tmp in ("csv", "xlsx"):
            try:
                if ext_tmp == "csv":
                    df_tmp = pd.read_csv(io.BytesIO(file_bytes_tmp))
                else:
                    df_tmp = pd.read_excel(io.BytesIO(file_bytes_tmp), engine="openpyxl")
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
if uploaded is not None:
    ext = _file_ext(uploaded.name)
    file_bytes = st.session_state.get("_uploaded_bytes") or uploaded.getvalue()

    # Spreadsheet size limit
    if ext in ("csv", "xlsx") and len(file_bytes) > 5 * 1024 * 1024:
        st.error("File too large. Please upload a CSV/XLSX up to 5 MB.")
        st.stop()

    llm = get_gemini(model="gemini-2.5-flash", temperature=0.1)

    if ext == "pdf":
        # Show processing indicator and disable inputs while building the index
        if st.session_state.get("file_id") != hashlib.md5(file_bytes).hexdigest() or st.session_state.get("vectorstore") is None:
            st.session_state.doc_processing = True
            with st.status("Processing document…", expanded=False) as status:
                is_new = ensure_pdf_vectorstore(uploaded.name, file_bytes)
                status.update(label="Document ready", state="complete")
            st.session_state.doc_processing = False
        else:
            is_new = False
        vectorstore = st.session_state.vectorstore
        if is_new:
            st.info("Previous chat cleared — ready for new document analysis.")

        # Minimal doc label near header
        doc_label = st.session_state.get('doc_title') or uploaded.name
        if st.session_state.get("page_count"):
            st.caption(f"Document: {doc_label} • Pages: {st.session_state.page_count}")
        else:
            st.caption(f"Document: {doc_label}")

        if mode == "Summarize":
            with st.status("Summarizing document…", expanded=False) as status:
                try:
                    seeds = ["overview", "introduction", "method", "results", "conclusion"]
                    seen = set()
                    pool: List[Document] = []
                    for q in seeds:
                        docs = search_documents(vectorstore, q, k=4, method=st.session_state.retrieval_method)
                        for d in docs:
                            key = hash((d.page_content[:256], d.metadata.get("page")))
                            if key not in seen:
                                seen.add(key)
                                pool.append(d)
                        if len(pool) >= max(10, st.session_state.k_results):
                            break

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
- Do not include bullet points or invented facts.

Document content:
{context}
"""
                    answer = llm_invoke(llm, prompt)
                except Exception as e:
                    answer = f"Error during summarization: {e}"
                status.update(label="Summary ready", state="complete")
            st.markdown(answer)
        elif mode == "Extract Fields":
            text = chunks_to_full_text(st.session_state.get("_raw_chunks") or [])
            if len(text) < 20:
                # Try OCR pass if text is too short (process_pdf already tries OCR, but double-check)
                try:
                    from pdf2image import convert_from_path  # type: ignore
                    import pytesseract  # type: ignore
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name
                    images = convert_from_path(tmp_path)
                    ocr_texts = []
                    for img in images:
                        try:
                            ocr_texts.append(clean_text(pytesseract.image_to_string(img)))
                        except Exception:
                            continue
                    text = "\n".join([t for t in ocr_texts if t])
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                except Exception:
                    pass

            if not text:
                st.error("Could not extract text from this PDF (install Tesseract/Poppler for OCR).")
            else:
                kv = extract_from_text(text)
                # Minimal UI: only a two-column table
                df_key = pd.DataFrame([{"Field Name": k, "Extracted Value": v} for k, v in kv.items()]) if kv else pd.DataFrame(columns=["Field Name", "Extracted Value"])
                st.dataframe(df_key, use_container_width=True, hide_index=True)
                # Optional download buttons (silent UI)
                csv_bytes = df_key.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv_bytes, file_name="extracted_fields.csv", mime="text/csv")
                out = io.BytesIO()
                with pd.ExcelWriter(out, engine="openpyxl") as xw:
                    df_key.to_excel(xw, index=False, sheet_name="ExtractedFields")
                st.download_button("Download Excel", data=out.getvalue(), file_name="extracted_fields.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            # Chat
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            if st.session_state.doc_processing:
                st.info("Processing document… chat will be available shortly.")
                user_q = None
            else:
                user_q = st.chat_input("Ask about this document…")
            if user_q:
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)
                with st.spinner("Thinking..."):
                    try:
                        # Use robust retrieval
                        docs = robust_retrieve(
                            vectorstore,
                            user_q,
                            k=st.session_state.k_results,
                            method=st.session_state.retrieval_method,
                            chunks=st.session_state.get("_raw_chunks"),
                        )
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
                        prompt = f"""
Use the provided context to answer the question. If not found, say you cannot find it in the document.

Context:
{context}

Question: {user_q}
"""
                        answer = llm_invoke(llm, prompt)
                        if st.session_state.show_sources and docs:
                            labels = []
                            for d in docs:
                                meta = d.metadata or {}
                                if meta.get("page"):
                                    labels.append(str(meta.get("page")))
                                elif meta.get("page_start") and meta.get("page_end"):
                                    labels.append(f"{meta['page_start']}–{meta['page_end']}")
                            if labels:
                                answer += f"\n\nSources: pages {', '.join(sorted(set(labels)))}"
                    except Exception as e:
                        answer = f"Error: {e}"
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)

    elif ext in ("jpg", "jpeg", "png"):
        # Extract Fields for images
        # Minimal file label near header
        st.caption(f"Image: {uploaded.name}")

        img_text = ocr_image_bytes(file_bytes)
        if not img_text:
            st.error("Could not OCR this image. Install pytesseract or easyocr and try again.")
        else:
            kv = extract_from_text(img_text)
            df_key = pd.DataFrame([{"Field Name": k, "Extracted Value": v} for k, v in kv.items()]) if kv else pd.DataFrame(columns=["Field Name", "Extracted Value"])
            st.dataframe(df_key, use_container_width=True, hide_index=True)
            csv_bytes = df_key.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="extracted_fields.csv", mime="text/csv")
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as xw:
                df_key.to_excel(xw, index=False, sheet_name="ExtractedFields")
            st.download_button("Download Excel", data=out.getvalue(), file_name="extracted_fields.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif ext in ("csv", "xlsx"):
        # Spreadsheet analytics
        try:
            if ext == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        except Exception as e:
            st.error(f"Failed to read spreadsheet: {e}")
            st.stop()
        st.session_state.df = df

        # Reset chat/context if different spreadsheet content
        file_id = hashlib.md5(file_bytes).hexdigest()
        if st.session_state.get("file_id") != file_id:
            st.session_state.messages = []
            st.session_state.last_context_docs = []
            st.session_state.last_context_query = None
            st.session_state.last_context_file_id = None
            st.session_state.file_id = file_id
            st.info("Previous chat cleared — ready for new document analysis.")

        # Minimal file label near header
        st.caption(f"Spreadsheet: {uploaded.name} • Rows: {df.shape[0]} • Columns: {df.shape[1]}")

        insights = analyze_spreadsheet(df)
        st.markdown(insights)
        st.dataframe(df.head(15), use_container_width=True)

        # Simple visuals
        try:
            def _ok(c: str) -> bool:
                s = df[c].astype(str)
                nuniq = s.nunique(dropna=True)
                if not (1 < nuniq <= 30):
                    return False
                bad = ("name", "email", "id")
                return not any(b in c.lower() for b in bad)

            cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if _ok(c)]
            selected = st.session_state.get("selected_category_col")
            options_inline = ["(auto)"] + cat_cols
            default_idx = options_inline.index(selected) if selected in options_inline else 0
            chosen = st.selectbox("Category for bar chart", options_inline, index=default_idx, key="cat_select_inline")
            if chosen != "(auto)":
                st.session_state.selected_category_col = chosen
                selected = chosen

            cat_col = selected if selected and selected != "(auto)" and selected in df.columns else (cat_cols[0] if cat_cols else None)
            if cat_col:
                vc = df[cat_col].astype(str).value_counts().head(10)
                st.subheader(f"Top categories in {cat_col}")
                st.bar_chart(vc)
        except Exception:
            pass

        if mode == "Summarize":
            with st.status("Summarizing dataset…", expanded=False) as status:
                try:
                    context = df_profile_snippet(df, max_cols=12, max_rows=6)
                    prompt = f"""
You are a data analyst. Summarize the dataset described below in 3–5 sentences.
Include key columns, notable distributions, missing data, and high-level insights.
Avoid hallucinations.

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
            user_q = st.chat_input("Ask about this dataset…")
            if user_q:
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)
                with st.spinner("Thinking..."):
                    try:
                        structured = answer_dataset_question(df, user_q)
                        if structured is not None:
                            answer = structured
                        else:
                            context = df_profile_snippet(df, max_cols=12, max_rows=6)
                            prompt = f"""
Use the dataset profile and sample rows to answer the user's question.
If the answer isn't apparent from the provided data, say so.

Dataset profile:
{context}

Question: {user_q}
"""
                            answer = llm_invoke(llm, prompt)
                    except Exception as e:
                        answer = f"Error: {e}"
                if answer:
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
    else:
        st.error("Unsupported file type. Please upload a PDF, CSV/XLSX, or an image (JPG/PNG).")
else:
    # Memory-only chat
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    user_q = st.chat_input("Ask anything… (no document loaded)")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.spinner("Thinking..."):
            try:
                hist_msgs = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")][-6:]
                history = []
                for i in range(0, len(hist_msgs), 2):
                    try:
                        qh = hist_msgs[i]["content"]
                        ah = hist_msgs[i + 1]["content"]
                        history.append(f"User: {qh}\nAssistant: {ah}")
                    except Exception:
                        pass
                history_text = "\n".join(history)
                prompt = f"""
You are a helpful assistant. There is no document loaded. Answer using your general knowledge. Use the conversation for additional context if helpful.

Conversation (recent, may be empty):
{history_text}

User question: {user_q}
"""
                llm = get_gemini(model="gemini-2.5-flash", temperature=0.2)
                answer = llm_invoke(llm, prompt)
            except Exception as e:
                answer = f"Error: {e}"
        if answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
