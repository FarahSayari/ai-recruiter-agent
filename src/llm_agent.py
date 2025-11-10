from __future__ import annotations
import os
from typing import Any, Dict, List

try:
    from .agentGraph import run_agent
except ImportError:
    from agentGraph import run_agent

# --- replaced LangChain wrappers with direct Ollama client ---
try:
    from ollama import Client
except ImportError:
    Client = None


class RecruiterAgent:
    def __init__(self, model: str | None = None, temperature: float = 0.2):
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3")
        self.temperature = temperature  # retained for future tuning (ollama ignores for now)
        self.client = self._init_client()

    def _init_client(self):
        if Client is None:
            raise RuntimeError("Install the 'ollama' package: pip install ollama")
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return Client(host=host)

    def _ask(self, prompt: str) -> str:
        try:
            resp = self.client.generate(model=self.model, prompt=prompt)
            return ((resp or {}).get("response", "") or "").strip().strip('"').strip("'")
        except Exception as e:
            return f"LLM error: {e}"

    # --- updated: deterministic heuristics first, LLM fallback second ---
    def _detect_intent(self, message: str) -> str:
        msg_l = (message or "").lower().strip()

        # Greeting
        if msg_l in {"hi", "hello", "hey"} or msg_l.startswith(("hi ", "hello ", "hey ")):
            return "greeting"

        # Upload CV
        if "upload" in msg_l and ("cv" in msg_l or "resume" in msg_l):
            return "upload_cv"

        # Search candidates heuristics
        search_triggers = ["find", "search", "source", "look for", "looking for", "need", "hire", "hiring", "recruit"]
        role_terms = [
            "engineer","developer","scientist","analyst","manager","architect",
            "devops","data","ml","ai","designer","intern","lead","full stack",
            "frontend","back end","backend","software","sre","platform"
        ]
        if any(t in msg_l for t in search_triggers) and (any(r in msg_l for r in role_terms) or " with " in msg_l or " and " in msg_l):
            return "search_candidates"

        # LLM fallback classification (kept brief and strict)
        prompt = f"""
Return only one label on a single line:
search_candidates
upload_cv
greeting
unknown

User: {message}
Label:
""".strip()
        label = self._ask(prompt).lower()
        if "search_candidates" in label or label.strip() in {"search", "search candidates"}:
            return "search_candidates"
        if "upload_cv" in label or ("upload" in msg_l and ("cv" in msg_l or "resume" in msg_l)):
            return "upload_cv"
        if "greeting" in label or msg_l in {"hi", "hello", "hey"}:
            return "greeting"
        return "unknown"

    def _format_candidates(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No strong matches yet. Try refining the role, must‑have skills, or seniority."
        lines: List[str] = ["Here are the top recommendations:"]
        for i, r in enumerate(results, start=1):
            cid = r.get("candidate_id") or r.get("id") or f"candidate-{i}"
            expl = (r.get("explanation") or "").strip()
            if not expl:
                expl = "Highly recommended for this role due to strong alignment with the job requirements and relevant experience."
            # Clean up explanation
            expl = " ".join(expl.split())
            if not expl.endswith((".", "!", "?")):
                expl += "."
            lines.append(f"{i}. {cid}: {expl}")
        return "\n".join(lines)

    def handle_message(self, message: str) -> str:
        msg = (message or "").strip()
        if not msg:
            return "Hi, I'm your AI recruiter. What role or skills are you looking for?"

        intent = self._detect_intent(msg)

        if intent == "search_candidates":
            # Always request explanations from the pipeline
            try:
                results = run_agent(job_desc=msg, top_k=5, with_explain=True)
            except TypeError:
                try:
                    results = run_agent(msg, top_k=5, with_explain=True)
                except TypeError:
                    results = run_agent(msg)
            return self._format_candidates(results)

        if intent == "greeting":
            return "Hi, I'm your AI recruiter. How can I help you today?"

        if intent == "upload_cv":
            return "Please upload the CV via the app UI and tell me the role to match."

        # Short, non-verbose reply for general/unknown
        prompt = f"Reply briefly (one sentence) as a helpful recruiter to: {msg}"
        return self._ask(prompt)
