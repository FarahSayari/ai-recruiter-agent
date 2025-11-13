import streamlit as st
import requests
import os
from io import BytesIO

# ----------------- Config -----------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
st.set_page_config(page_title="Aura – AI Recruiting Assistant", page_icon="🎯", layout="wide")

# ----------------- Custom CSS (Gradient Pink/Violet Style) -----------------
st.markdown("""
<style>
body {
    background-color: #ffffff;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* Remove Streamlit default padding */
.main > div { padding-top: 1rem !important; padding-bottom: 0 !important; }

/* Chat container */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem 1rem 8rem 1rem;
    overflow-y: auto;
}

/* Messages */
.message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1.1rem;
}

.message.user {
    justify-content: flex-end;
}

.bubble {
    border-radius: 12px;
    padding: 0.8rem 1rem;
    max-width: 80%;
    line-height: 1.45;
    font-size: 15px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    white-space: pre-wrap;
}

/* Agent bubble (light gray) */
.message.agent .bubble {
    background-color: #f7f7f8;
    color: #111827;
}

/* User bubble (gradient pink → violet) */
.message.user .bubble {
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    color: white;
}

/* Header / Aura intro */
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header p {
    color: #6b7280;
    font-size: 0.95rem;
}

/* Input bar (fixed bottom like ChatGPT) */
.input-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #ffffff;
    border-top: 1px solid #e5e7eb;
    padding: 0.8rem 0;
}

.input-container {
    max-width: 800px;
    margin: 0 auto;
    display: flex;
    gap: 0.5rem;
    align-items: center;
}

/* Text input */
.stTextInput>div>div>input {
    border-radius: 24px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.7rem 1rem !important;
    font-size: 15px !important;
}

/* Gradient button */
.stButton>button {
    border-radius: 24px !important;
    background: linear-gradient(135deg, #ec4899, #8b5cf6) !important;
    color: white !important;
    font-weight: 500;
    padding: 0.6rem 1rem;
    font-size: 15px;
    border: none !important;
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #f472b6, #a78bfa) !important;
    transform: scale(1.02);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.title("Aura")
    api_base = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    if st.button("New Chat", use_container_width=True):
        st.session_state["messages"] = [{"role": "agent", "content": "Ready when you are."}]
        st.rerun()

# ----------------- Session State -----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "agent", "content": "Ready when you are."}]

# Deferred clearing mechanism (avoid modifying widget key after instantiation in same run)
if st.session_state.get("clear_input"):
    # Perform clear BEFORE rendering widgets
    st.session_state["user_input"] = ""
    del st.session_state["clear_input"]
    # Increment uploader version to force a fresh empty uploader widget
    st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1

if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0

# ----------------- Helper -----------------
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def call_chat_api(message, file_bytes=None, filename=None):
    try:
        if file_bytes and filename:
            files = {"file": (filename, BytesIO(file_bytes), "application/pdf")}
            data = {"message": message}
            resp = requests.post(API_URL, files=files, data=data, timeout=240)
        else:
            payload = {"message": message}
            resp = requests.post(API_URL, json=payload, timeout=180)

        if resp.status_code != 200:
            return False, f"API error {resp.status_code}: {resp.text}"
        data = resp.json()
        return True, data.get("reply", resp.text)
    except Exception as e:
        return False, f"Request failed: {e}"

# ----------------- Chat UI -----------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Show Aura intro only before first user message
if not any(m["role"] == "user" for m in st.session_state.messages):
    st.markdown("""
    <div class="header">
        <h1>Aura</h1>
        <p>Ready when you are.</p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for m in st.session_state.messages:
    role = m["role"]
    content = m["content"]
    if role == "agent" and content.lower() == "ready when you are.":
        continue
    st.markdown(
        f'<div class="message {role}"><div class="bubble">{content}</div></div>',
        unsafe_allow_html=True
    )

# Smooth auto-scroll to bottom after messages
st.markdown("""
<script>
var chatContainer = document.getElementsByClassName('chat-container')[0];
if(chatContainer){
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: 'smooth' });
}
</script>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Input Bar -----------------
st.markdown('<div class="input-bar">', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([8, 1])
    with col1:
        # Initialize session key if absent, then use key-only pattern (no explicit value) to allow clearing safely
        if "user_input" not in st.session_state:
            st.session_state["user_input"] = ""
        st.text_input("Message", placeholder="Message Aura...", label_visibility="collapsed", key="user_input")
    with col2:
        send_clicked = st.button("Send", use_container_width=True)
    # Key for uploader allows resetting via st.session_state
    uploader_key = f"job_pdf_{st.session_state['uploader_version']}"
    uploaded_file = st.file_uploader("Upload job description PDF", type=["pdf"], label_visibility="collapsed", key=uploader_key)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------- Handle Send -----------------
if send_clicked:
    msg = (st.session_state.get("user_input", "") or "").strip()
    file_obj = uploaded_file  # use direct variable, not session state
    if not msg and not file_obj:
        st.warning("Please type a message or upload a file.")
    else:
        add_message("user", msg or "(PDF uploaded)")
        file_bytes = file_obj.read() if file_obj else None
        filename = file_obj.name if file_obj else None

        with st.spinner("Thinking..."):
            ok, reply = call_chat_api(msg, file_bytes, filename)
        add_message("agent", reply if ok else f"⚠️ {reply}")

    # Flag for clearing on next run & trigger uploader reset
    st.session_state["clear_input"] = True
    st.rerun()
