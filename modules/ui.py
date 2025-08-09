import streamlit as st


def set_page():
    st.set_page_config(page_title="Atlas: AI Doc & Data Companion", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
            .main {padding-top: 1rem;}
            h1, h2, h3 {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;}
            .status {color:#6b7280; font-size:0.9rem}
            .pill {display:inline-block; padding:2px 8px; border-radius:9999px; background:#eef2ff; color:#4338ca; margin-right:6px;}
            .tool-card {background:#0b0f19; border:1px solid #1f2937; padding:12px; border-radius:10px;}
            .muted {color:#9ca3af}
            /* Gradient title styling */
            .app-title { 
                font-size: 2.1rem; 
                font-weight: 800; 
                letter-spacing: .2px;
                background: linear-gradient(90deg, #22d3ee, #a78bfa 45%, #f472b6);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin: 0 0 .25rem 0;
            }
            .app-subtitle { color: #6b7280; margin-bottom: .75rem; }
            [data-testid="stSidebar"] > div:first-child {
                position: sticky;
                top: 0;
                height: 100vh;
                overflow: auto; /* allow content to be accessible */
            }
            /* Subtle cards */
            .card {background: rgba(2,6,23,.6); border:1px solid #111827; padding:14px; border-radius:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, status_text: str):
    st.markdown(f"<div class='app-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status'>{status_text}</div>", unsafe_allow_html=True)


def app_banner():
    left, right = st.columns([4, 1])
    with left:
        st.markdown("<div class='app-title'>Atlas</div>", unsafe_allow_html=True)
        st.markdown("<div class='app-subtitle'>Your local AI companion for PDFs and spreadsheets</div>", unsafe_allow_html=True)
        st.markdown("<span class='pill'>RAG</span><span class='pill'>Summarize</span><span class='pill'>Chat</span>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='status'>Local • Private</div>", unsafe_allow_html=True)
