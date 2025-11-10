from __future__ import annotations
import os
import re
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
        self.chat_top_k = int(os.getenv("CHAT_TOP_K", "3"))  # reduce LLM calls for chat

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

    # --- new: infer desired top_k from natural language message ---
    def _infer_top_k(self, text: str, default: int | None = None) -> int:
        t = (text or "").lower()
        default_k = default if default is not None else self.chat_top_k

        # Direct patterns: "top 3", "best 2", "top-5"
        m = re.search(r"(top|best)\s*[- ]?\s*(\d+)", t)
        if m:
            try:
                k = int(m.group(2))
                return min(max(k, 1), 10)
            except Exception:
                pass

        # "best candidate" or "the best candidate"
        if "best candidate" in t or ("best" in t and "candidate" in t and re.search(r"\b(best)\b.*\bcandidate(s)?\b", t)):
            return 1

        # Generic number in context of candidates
        m2 = re.search(r"\b(\d+)\s+(candidates|people|profiles|engineers|developers|scientists)\b", t)
        if m2:
            try:
                k = int(m2.group(1))
                return min(max(k, 1), 10)
            except Exception:
                pass

        # Number words
        words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "couple": 2, "few": 3, "single": 1
        }
        for w, val in words.items():
            if re.search(rf"\b{w}\b\s+(candidate|candidates)", t) or re.search(rf"(top|best)\s+{w}", t):
                return min(max(val, 1), 10)

        return default_k

    # Public helper if API needs it
    def infer_top_k_from_text(self, text: str) -> int:
        return self._infer_top_k(text, self.chat_top_k)

    def _format_candidates(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No strong matches yet. Try refining the role, must‑have skills, or seniority."
        lines: List[str] = ["Here are the top recommendations:"]
        for i, r in enumerate(results, start=1):
            cid = r.get("candidate_id") or r.get("id") or f"candidate-{i}"
            expl = (r.get("explanation") or "").strip()
            if not expl:
                expl = "Highly recommended for this role due to strong alignment with the job requirements and relevant experience."
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
            k = self._infer_top_k(msg, self.chat_top_k)
            try:
                results = run_agent(job_desc=msg, top_k=k, with_explain=True)
            except TypeError:
                try:
                    results = run_agent(msg, top_k=k, with_explain=True)
                except TypeError:
                    results = run_agent(msg)
            return self._format_candidates(results)

        if intent == "greeting":
            return "Hi, I'm your AI recruiter. How can I help you today?"

        if intent == "upload_cv":
            return "Please upload the CV via the app UI and tell me the role to match."

        prompt = f"Reply briefly (one sentence) as a helpful recruiter to: {msg}"
        return self._ask(prompt)

    # --- direct handler for job description text (chat-style output) ---
    def handle_job_description(self, job_desc: str, top_k: int = 5) -> str:
        jd = (job_desc or "").strip()
        if not jd:
            return "Please provide a job description."
        try:
            results = run_agent(job_desc=jd, top_k=top_k, with_explain=True)
        except TypeError:
            try:
                results = run_agent(jd, top_k=top_k, with_explain=True)
            except TypeError:
                results = run_agent(jd)
        return self._format_candidates(results)
