<div align="center">
<h1>Aura – AI Recruiter Agent</h1>
<p><strong>Source, rank, explain and contact top candidates from your own CV / resume datastore.</strong></p>
<p>Streamlit UI · FastAPI backend · Qdrant vector search · BERT embeddings · Ollama LLM (optional) · SMTP scheduling & outreach</p>
</div>

---

## Table of Contents
1. Overview
2. Features
3. Architecture
4. Quick Start (Windows / PowerShell)
5. Environment Variables
6. Usage Flow
7. Candidate Formatting Rules
8. Scheduling & Emailing
9. Performance Tuning
10. FAQ (Built‑in Instant Answers)
11. Extending the Agent
12. Security & Privacy Notes
13. Troubleshooting

---

## 1. Overview
Aura is an AI recruiting assistant that ingests / indexes cleaned resumes (CVs), then lets you:
- Provide a job description (paste or PDF upload)
- Retrieve and rank the top N matching candidates
- Generate concise skill/experience/achievement rationales (2–3 sentences)
- Schedule interview slots and send invitation emails
- Ask built‑in FAQ questions (“Who are you?”, “What can you do?”, etc.)

The system favors speed and determinism: per‑candidate explanations are cached / optionally skipped; top_k is enforced strictly; output format is stable.

---

## 2. Features
| Area | Capability |
|------|------------|
| Retrieval | Vector search over resumes via Qdrant + BERT embeddings |
| Ranking | Qdrant score (fast) or optional local cosine re‑ranking |
| Explanation | LLM (Ollama) 2–3 sentence candidate fit rationale (skippable) |
| Formatting | Name‑first, numbered list, no internal IDs, consistent length |
| Job Description | Text input or PDF upload (auto‑clean & embed) |
| Email Outreach | SMTP invites (dry‑run or real) with interview scheduling |
| Scheduling | Auto next business day 30‑min slots with meeting links |
| FAQ | Instant canned answers (no LLM latency) |
| Performance Toggles | Disable explanations, disable local rerank, limit tokens |
| Caching | LRU cache for job description embeddings & candidate phrasing |

---

## 3. Architecture
```
Streamlit (UI)  -->  FastAPI /chat, /query, /schedule_interviews, /send_emails
					    |
					    v
				 RecruiterAgent (intent, formatting, FAQ)
					    |
				  Agent Graph (retrieve -> analyze -> skills -> rank -> explain)
					    |
				    Qdrant (vectors + payload email/name)
					    |
				   BERT Embeddings (transformers)
					    |
				    Ollama LLM (phi3 or configured) [optional]
					    |
					SMTP (invite emails)
```
Key files:
- `src/app.py` – Streamlit interface
- `src/api.py` – FastAPI endpoints (`/chat`, `/query`, `/schedule_interviews`, `/send_emails`)
- `src/llm_agent.py` – Intent detection, FAQ, formatting
- `src/agentGraph.py` – LangGraph‑style pipeline + scheduling, emailing
- `src/agentNodes.py` – Low‑level nodes (analyze, rank, explain)
- `src/storage.py` – Qdrant storage abstraction
- `src/embeddings.py` – BERT embedding generation
- `src/mailer.py` – SMTP send utility

---

## 4. Quick Start (Windows / PowerShell)
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

## 5. Environment Variables
Set in PowerShell (session) before starting services:
```powershell
$env:API_URL = "http://127.0.0.1:8000/chat"            # Frontend API base
$env:OLLAMA_HOST = "http://localhost:11434"            # Ollama server
$env:OLLAMA_MODEL = "phi3"                              # Or fast model (phi3:mini, llama3.2:1b-instruct)
$env:CHAT_TOP_K = "3"                                   # Default results in chat
$env:EXPLAIN_WITH_LLM = "1"                            # Set "0" to skip LLM explanations (fast mode)
$env:RERANK_LOCALLY = "0"                               # Set "1" to fetch vectors & cosine rerank
$env:LLM_NUM_PREDICT = "128"                            # Limit tokens for faster generation
$env:MAX_RESUME_CHARS = "1200"                          # Prompt truncation
$env:MAX_JOB_CHARS = "800"                              # Prompt truncation

# SMTP (Gmail example – use App Password)
$env:SMTP_HOST = "smtp.gmail.com"
$env:SMTP_PORT = "587"
$env:SMTP_USER = "your_email@gmail.com"
$env:SMTP_PASS = "your_16char_app_password"            # NEVER commit this
$env:FROM_EMAIL = "your_email@gmail.com"
$env:REPLY_TO = "your_email@gmail.com"
$env:EMAIL_DRY_RUN = "1"                               # Set "0" to actually send
```
Persistent (across sessions) alternative: `setx VAR "value"` (then restart shell).

---

## 6. Usage Flow
1. Launch Streamlit & API.
2. Paste or upload a PDF job description.
3. Ask: “Find top 3 candidates for this JD.”
4. Review concise candidate list (name‑first, 2–3 sentences, score).
5. Request scheduling: “Book appointments and send them emails.”
   - Response (configured to concise format): `Scheduled and attempted emails for N candidates.`
6. Ask built‑in FAQs any time (“Who are you?”, “What can you do?”, etc.).

---

## 7. Candidate Formatting Rules
Implemented in `RecruiterAgent._format_candidates`:
- Strict top_k enforcement (never more than requested). 
- Each candidate: 2–3 sentences; no ellipses or truncation markers.
- Focus on skills, experience, action verbs (built, led, implemented etc.).
- Name extracted heuristically from resume / explanation; fallback to “Candidate”.
- No internal candidate_id shown to user.
- Optional `(score X.XX)` appended.

---

## 8. Scheduling & Emailing
Workflow in `agentGraph.py`:
- `run_agent_schedule()` loads last search state.
- Generates next business day 30‑minute interview slots with links.
- Invokes email sender node; uses real emails from Qdrant payload / resume text.
- Returns summary: `Scheduled and attempted emails for N candidates.`

Dry‑run mode (`EMAIL_DRY_RUN=1`) logs payload without sending; set `0` to send.

---

## 9. Performance Tuning
| Toggle | Effect |
|--------|--------|
| EXPLAIN_WITH_LLM=0 | Skip per‑candidate LLM explanation calls (largest speed gain) |
| RERANK_LOCALLY=0 | Trust Qdrant score; prevents vector transfer & cosine calc |
| LLM_NUM_PREDICT | Cap generation tokens |
| CHAT_TOP_K | Lower candidate count for chat (default 3) |
| MAX_RESUME_CHARS / MAX_JOB_CHARS | Smaller prompts, faster inference |

Other tips:
- Use a smaller Ollama model (phi3:mini or llama3.2:1b-instruct) for faster explanations.
- Pre‑embed & store resume email in Qdrant payload (avoid regex fallback).
- Keep BERT model on GPU if available (transformers auto‑detect). 

---

## 10. FAQ (Instant Answers)
Aura answers these instantly (no LLM latency):
| Question | Answer Summary |
|----------|----------------|
| Who are you? | Aura, AI recruiting assistant; source, rank, contact candidates. |
| What can you do? | Extract JD skills, find top N, explain picks, draft outreach, schedule interviews. |
| What data do you use? | Your indexed resumes/CVs + job info you provide only. |
| How do you rank candidates? | Score resume skills/experience vs. JD; list top N with reasons. |
| How should I give you a job description? | Paste or upload PDF; ask “find top 3 candidates for this JD”. |

Extend easily by adding patterns in `_faq_reply()` inside `llm_agent.py`.

---

## 11. Extending the Agent
Ideas:
- Add “Compare <Name1> vs <Name2>” intent with side‑by‑side strengths/gaps.
- Export shortlist as CSV (`/export` endpoint or Streamlit download button).
- Add a “Fast mode” toggle in UI flipping `EXPLAIN_WITH_LLM`.
- Integrate salary/range filters (payload metadata) for queries.
- Introduce a lightweight NER for more reliable candidate name extraction.

---

## 12. Security & Privacy Notes
- Never commit SMTP credentials or App Passwords; use environment variables.
- Emails are only sent when `EMAIL_DRY_RUN=0`.
- Resume data stays local (Qdrant + your filesystem). No external calls besides Ollama & SMTP.
- Consider encryption / access control for production deployments.

---

## 13. Troubleshooting
| Symptom | Cause | Fix |
|---------|-------|-----|
| Streamlit input not clearing | Widget state mutation race | Uses deferred `clear_input` + uploader version key |
| Missing candidate names | Heuristic fails | Store explicit `name` in Qdrant payload during ingestion |
| Emails not sent | EMAIL_DRY_RUN=1 or bad SMTP creds | Set `EMAIL_DRY_RUN=0`, verify App Password |
| Slow responses | LLM explanations + rerank | Set `EXPLAIN_WITH_LLM=0`, `RERANK_LOCALLY=0`, lower `CHAT_TOP_K` |
| 500 on /chat | Exception in retrieval or LLM | Check API logs; verify Ollama running & Qdrant reachable |
| Import error langgraph | Optional graph module missing | Install `langgraph` or rely on simple fallback inside `agentGraph.py` |

Logs / timing: FastAPI adds `X-Process-Time-ms` header; inspect for latency comparisons.

---

## License
This project currently has no explicit license file. Add one (e.g., MIT) if you plan to distribute.

---

## Contributing
Open to improvements: PRs welcome for better name extraction, additional intents, model adapters, or multi‑tenant storage.

---

## Disclaimer
This is a prototype AI assistant; always review candidates manually before making hiring decisions. Ensure compliance with local data protection and anti‑discrimination regulations.

---

Enjoy recruiting with Aura! 🚀
