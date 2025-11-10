import streamlit as st
import requests
import os
from io import BytesIO

# Resolve API URL without relying on Streamlit secrets
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")

# --------------- Page config & Styles ---------------
st.set_page_config(page_title="AI Recruiting Assistant", page_icon="🎯", layout="centered")

CUSTOM_CSS = """
<style>
/* Center title */
.center-title { text-align: center; margin-top: -1rem; }

/* Chat container */
.chat-container {
  max-width: 820px;
  margin: 0 auto;
  border: 1px solid #e6e6e6;
  border-radius: 12px;
  padding: 12px 12px 0 12px;
  height: 60vh;
  overflow-y: auto;
  background: #ffffff;
}
.msg { display: flex; margin: 8px 0; }
.msg.user { justify-content: flex-end; }
.msg.agent { justify-content: flex-start; }
.bubble {
  padding: 10px 14px;
  border-radius: 14px;
  max-width: 80%;
  line-height: 1.4;
  white-space: pre-wrap;
}
.bubble.user {
  background: #f0f0f0;
  color: #111;
}
.bubble.agent {
  background: #e8f0fe; /* light blue */
  color: #0b3d91;
}
.input-row {
  max-width: 820px;
  margin: 10px auto 0 auto;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------- Sidebar (optional settings) ---------------
with st.sidebar:
    st.markdown("### Settings")
    api_base = st.text_input("API URL", value=API_URL, help="FastAPI /chat endpoint URL")
    st.caption("Set API_URL env var to change default.")

# --------------- Session State ---------------
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "agent", "content": "Hi, I’m your AI Recruiting Assistant. How can I help you today?"}
    ]

def add_message(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})


# --------------- Title ---------------
st.markdown('<h1 class="center-title">🎯 AI Recruiting Assistant</h1>', unsafe_allow_html=True)

# --------------- Chat Container ---------------
chat_placeholder = st.container()
with chat_placeholder:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for m in st.session_state.chat:
        role = m.get("role")
        cls = "user" if role == "user" else "agent"
        content = m.get("content", "")
        st.markdown(f'<div class="msg {cls}"><div class="bubble {cls}">{content}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------- Input Area ---------------
st.markdown('<div class="input-row">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])
with col1:
    user_text = st.text_input("Message", value="", placeholder="Type a message (e.g., Find top 3 data scientists with NLP) ...")
with col2:
    send_clicked = st.button("Send", type="primary", use_container_width=True)

uploaded_file = st.file_uploader("Upload job description PDF (optional)", type=["pdf"], accept_multiple_files=False)
st.markdown('</div>', unsafe_allow_html=True)


def call_chat_api(message: str, file_bytes: bytes | None, filename: str | None) -> tuple[bool, str]:
    """Return (ok, reply_text). Sends multipart if file present; else JSON."""
    headers = {}

    try:
        if file_bytes is not None and filename:
            files = {
                "file": (filename, BytesIO(file_bytes), "application/pdf"),
            }
            data = {"message": message}
            # Allow more time for PDF parsing + retrieval
            resp = requests.post(api_base, files=files, data=data, headers=headers, timeout=240)
        else:
            payload = {"message": message}
            # Allow more time for model/explanations on slower machines
            resp = requests.post(api_base, json=payload, headers=headers, timeout=180)

        if resp.status_code != 200:
            return False, f"API error {resp.status_code}: {resp.text}" 
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        reply = (data or {}).get("reply")
        if not reply:
            # Some endpoints might return plain text
            reply = resp.text or "(no reply)"
        return True, reply
    except Exception as e:
        return False, f"Request failed: {e}"


# --------------- Handle submission ---------------
if send_clicked:
    msg = (user_text or "").strip()
    if not msg and not uploaded_file:
        st.warning("Please type a message or upload a PDF.")
    else:
        if msg:
            add_message("user", msg)
        else:
            add_message("user", "(PDF uploaded)")

        file_bytes = uploaded_file.read() if uploaded_file else None
        filename = uploaded_file.name if uploaded_file else None

        with st.spinner("Thinking..."):
            ok, reply = call_chat_api(msg, file_bytes, filename)

        if ok:
            add_message("agent", reply)
        else:
            add_message("agent", f"⚠️ {reply}")

        # Rerun to refresh the chat container (compat across Streamlit versions)
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()  # older versions
            except Exception:
                pass
