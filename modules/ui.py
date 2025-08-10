import streamlit as st


def set_page():
    # Minimal page config
    st.set_page_config(page_title="Buddy", page_icon="📑", layout="wide")
    # Compact, clean styles and minimal spacing
    st.markdown(
        """
        <style>
            .main { padding-top: 0.25rem; }
            h1, h2, h3 {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;}
            .status {color:#6b7280; font-size:0.9rem}
            .tool-card {background:#0b0f19; border:1px solid #1f2937; padding:12px; border-radius:12px; box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;}
            .muted {color:#9ca3af}
            /* Plain title styling (no gradient) */
            .app-title {
                font-size: 1.8rem;
                font-weight: 800;
                letter-spacing: .2px;
                color: #5f7199;
                margin: 0 0 .25rem 0;
            }
            /* Sidebar compact spacing */
            [data-testid="stSidebar"] .block-container { padding-top: .25rem; padding-bottom: .25rem; }
            [data-testid="stSidebar"] .stSlider, [data-testid="stSidebar"] .stCheckbox { margin-top: .25rem; margin-bottom: .25rem; }
            /* Minimize space between chat messages */
            [data-testid="stChatMessage"] { margin-bottom: .3rem; }
            /* Optional subtle cards */
            .card {background: rgba(2,6,23,.6); border:1px solid #111827; padding:14px; border-radius:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, status_text: str):
    st.markdown(f"<div class='app-title'>{title}</div>", unsafe_allow_html=True)
    if status_text and str(status_text).strip():
        st.markdown(f"<div class='status'>{status_text}</div>", unsafe_allow_html=True)


def app_banner():
    # Deprecated visual banner; keeping for backward compatibility (no-op minimalist header)
    header("Buddy", "")
