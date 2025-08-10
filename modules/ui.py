import streamlit as st


def set_page():
    st.set_page_config(page_title="Your Document Companion", page_icon="📑", layout="wide")
    st.markdown(
        """
        <style>
            .main {padding-top: 1rem;}
            h1, h2, h3 {font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;}
            .status {color:#6b7280; font-size:0.9rem}
            .pill {display:inline-block; padding:4px 10px; border-radius:9999px; background:linear-gradient(90deg,#eef2ff,#e0f2fe); color:#4338ca; margin-right:6px; border:1px solid #dbeafe}
            .tool-card {background:#0b0f19; border:1px solid #1f2937; padding:12px; border-radius:12px; box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset;}
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
            /* Banner container */
            .banner {background: linear-gradient(90deg, rgba(34,211,238,0.08), rgba(167,139,250,0.08)); border:1px solid #1f2937; padding:14px 16px; border-radius:14px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, status_text: str):
    st.markdown(f"<div class='app-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='status'>{status_text}</div>", unsafe_allow_html=True)


def app_banner():
    with st.container():
        left, right = st.columns([4, 2])
        with left:
            st.markdown("<div class='banner'><div class='app-title'>Your Document Companion</div>" \
                        "<div class='app-subtitle'>PDFs • Images • Spreadsheets</div>" \
                        "<span class='pill'>RAG</span><span class='pill'>Summarize</span><span class='pill'>Extract Fields</span><span class='pill'>Chat</span></div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='status' style='text-align:right;'>Ready</div>", unsafe_allow_html=True)
