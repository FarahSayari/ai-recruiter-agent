# Aura – AI Recruiter Agent

Automate sourcing, ranking, explaining, and contacting top candidates from a resume datastore using a multimodal AI agent powered by RAG, Qdrant, BERT embeddings, and LangGraph.

<p align="center">
  <img src="images/aura_newChat.png" width="750"/>
</p>

---

## Overview

**Aura** is an end-to-end AI Recruiting Assistant designed to reduce manual screening time for HR teams.  
It processes unstructured CVs (PDF, text, or images), cleans and embeds them, stores them in **Qdrant**, and uses a **RAG-driven multimodal agent** to:

- Parse, clean, and embed CVs into vector space  
- Match candidates to any job description  
- Retrieve & rank the **Top N** most relevant candidates  
- Generate short, explainable rationales (2–3 sentences)  
- Automate interview scheduling & email outreach  
- Answer built-in FAQs about how it works  

The agent is exposed through a **FastAPI endpoint** and consumed by a **Streamlit chat UI**.


---


## Screenshots & Examples
| Purpose | File |
|---------|------|
| Main UI (chat + input bar) | ![Aura UI](/images/aura_newChat.png) |
| FAQ | ![Aura UI](/images//1113.png) |
| Candidate list example | ![Aura UI](/images/1113(1).png) |
| Qdrant DB | ![Aura UI](/images/qdrant.png) |





---


Key files:
- `src/app.py` – Streamlit interface
- `src/api.py` – FastAPI endpoints (`/chat`)
- `src/llm_agent.py` – Intent detection, FAQ, formatting
- `src/agentGraph.py` – LangGraph‑style pipeline + scheduling, emailing
- `src/agentNodes.py` – Low‑level nodes (analyze, rank, explain)
- `src/storage.py` – Qdrant storage abstraction
- `src/embeddings.py` – BERT embedding generation
- `src/mailer.py` – SMTP send utility

---

## Quick Start (Windows / PowerShell)
```powershell
# 1. Create & activate virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run backend API
uvicorn src.api:app --reload --port 8000

# 4. In separate shell start Streamlit UI
cd src
streamlit run app.py
```
Open UI at http://localhost:8501 and API at http://localhost:8000/docs (Swagger).

---

## Environment Variables
Set in PowerShell (session) before starting services:
```powershell
# SMTP (Gmail example – use App Password)
$env:SMTP_HOST = "smtp.gmail.com"
$env:SMTP_PORT = "587"
$env:SMTP_USER = "your_email@gmail.com"
$env:SMTP_PASS = "your_16char_app_password"            # NEVER commit this
$env:FROM_EMAIL = "your_email@gmail.com"
$env:REPLY_TO = "your_email@gmail.com"
$env:EMAIL_DRY_RUN = "1"                               # Set "0" to actually send
```

---

## Usage Flow
1. Launch Streamlit & API.
2. Paste or upload a PDF job description.
3. Ask: “Find top 3 candidates for this JD.”
4. Review concise candidate list (name‑first, 2–3 sentences, score).
5. Request scheduling: “Book appointments and send them emails.”
   - Response (configured to concise format): `Scheduled and attempted emails for N candidates.`
6. Ask built‑in FAQs any time (“Who are you?”, “What can you do?”, etc.).

---


Other tips:
- Pre‑embed & store resume email in Qdrant payload (run main notebook cells).
- Keep BERT model on GPU if available (transformers auto‑detect). 



---

Enjoy recruiting with Aura! 🚀
