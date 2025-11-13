from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional

try:
    from .agentGraph import run_agent, run_agent_email, run_agent_schedule
except ImportError:
    from agentGraph import run_agent, run_agent_email, run_agent_schedule

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
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", str(self.chat_top_k)))

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

        # --- new: scheduling intent (calendar + email) ---
        schedule_triggers = [
            "schedule interviews", "schedule interview", "schedule them", "schedule for them",
            "book meetings", "book meeting", "book interviews", "book interview", "book appointments", "book appointment", "appointments",
            "calendar these", "set up meetings", "set appointments", "schedule appointments",
            "schedule with top candidates", "schedule for selected candidates",
        ]
        if any(phrase in msg_l for phrase in schedule_triggers):
            return "schedule_interviews"

        # --- email intent ---
        email_triggers = [
            "send emails", "send email", "email the candidates", "email candidates",
            "invite", "send invites", "schedule interviews", "interview emails",
            "send them interview", "send interview", "invite them"
        ]
        if any(phrase in msg_l for phrase in email_triggers):
            return "email_candidates"

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

    # --- improved: infer desired top_k from natural language message ---
    def _infer_top_k(self, text: str, default: Optional[int] = None) -> int:
        t = (text or "").lower()
        default_k = default if default is not None else self.chat_top_k

        # Ignore numbers tied to years/months (e.g., "3 years", "2+ years")
        t_wo_years = re.sub(r"\b\d+\s*\+?\s*(years?|yrs?|months?)\b", "", t)

        # Strong explicit patterns
        patterns = [
            r"top\s+(\d+)",
            r"top\s+(\d+)\s+(?:candidate|candidates|profiles)",
            r"(?:send|email|retrieve|get|show)\s+(?:the\s+)?(?:top\s+)?(\d+)\s+(?:candidate|candidates|profiles)",
            r"(\d+)\s+(?:best|top)\s+(?:candidate|candidates|profiles)",
            r"(?:candidate|candidates|profiles)\s*:\s*(\d+)",
        ]
        for pat in patterns:
            m = re.search(pat, t_wo_years)
            if m:
                try:
                    k = int(m.group(1))
                    return min(max(k, 1), 50)
                except Exception:
                    pass

        # Word numbers near candidate keywords
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "couple": 2, "few": 3, "single": 1
        }
        for w, val in number_words.items():
            if re.search(rf"\b{w}\b\s+(?:candidate|candidates|profiles)\b", t_wo_years) or re.search(rf"\b(top|best)\s+{w}\b", t_wo_years):
                return min(max(val, 1), 50)

        # Generic "[number] candidates"
        m2 = re.search(r"\b(\d+)\s+(?:candidate|candidates|profiles)\b", t_wo_years)
        if m2:
            try:
                k = int(m2.group(1))
                return min(max(k, 1), 50)
            except Exception:
                pass

        # "best candidate" implies 1
        if re.search(r"\bbest\s+candidate\b", t_wo_years):
            return 1

        # fallback
        return default_k

    # Public helper if API needs it
    def infer_top_k_from_text(self, text: str) -> int:
        return self._infer_top_k(text, self.chat_top_k)

    # --- new: instant FAQ replies (no LLM call) ---
    def _faq_reply(self, message: str) -> Optional[str]:
        t = (message or "").strip().lower()
        if not t:
            return None
        def has(*phrases: str) -> bool:
            return any(p in t for p in phrases)

        # Who are you?
        if has("who are you", "what are you", "who r u"):
            return (
                "I’m Aura, an AI recruiting assistant. I help you source, rank, and contact candidates based on your job requirements."
            )
        # What can you do?
        if has("what can you do", "what do you do", "how can you help"):
            return (
                "I can extract key skills from a JD, find and rank top N candidates, explain each pick in 2–3 sentences, draft outreach, and schedule interviews."
            )
        # What data do you use?
        if has("what data do you use", "what data you use", "what data do u use", "which data do you use"):
            return (
                "I search your indexed resumes/CVs and the job info you provide. I don’t use public data unless you share it."
            )
        # How do you rank candidates?
        if has("how do you rank", "how you rank candidates", "ranking candidates", "rank candidates how"):
            return (
                "I score skills and experience from the resume against the JD, then list the top N with concise reasons."
            )
        # How to provide JD?
        if has("how should i give you a job description", "how to give job description", "how to provide jd", "how to give jd", "upload a pdf"):
            return (
                "Paste the JD or upload a PDF. Then say something like: ‘find top 3 candidates for this JD’."
            )
        return None

    def _format_candidates(self, results: List[Dict[str, Any]]) -> str:
        """Return a concise, numbered list with 2–3 sentences per candidate.

        Rule implementation (per user spec):
        1. Numbered list of top N.
        2. 2–3 sentences each (cap words per sentence & total length).
        3. Focus on skills, experience, achievements (pull from explanation + skills).
        4. Filter generic filler (e.g. "good experience", "strong fit") and replace with specifics.
        5. Maintain consistency: reuse prior phrasing for same candidate in this agent instance.
        6. Keep overall answer concise.
        """
        if not results:
            return "No strong matches yet. Refine role, must‑have skills, or seniority keywords." 

        # Cache for consistency across calls in same session
        if not hasattr(self, "_candidate_cache"):
            self._candidate_cache: Dict[str, str] = {}

        max_words_sentence = int(os.getenv("REASON_SENTENCE_WORDS", "22"))
        max_sentences = 3
        max_total_chars = int(os.getenv("RESPONSE_MAX_CHARS", "2200"))

        generic_patterns = [
            r"\b(good|solid|strong|great) (experience|background|fit)\b",
            r"\bwell suited\b",
            r"\bfit for (the )?role\b",
            r"\bstrong match\b",
            r"\baligned with requirements\b",
        ]

        def clean_generic(text: str, skills: List[str]) -> str:
            if not text:
                return ""
            t = text
            for pat in generic_patterns:
                if re.search(pat, t, flags=re.IGNORECASE):
                    # Replace generic phrase with skill-based specificity
                    top_sk = ", ".join(skills[:3]) if skills else "role keywords"
                    t = re.sub(pat, f"demonstrated strength in {top_sk}", t, flags=re.IGNORECASE)
            return t

            
        def split_sentences(text: str) -> List[str]:
            t = (text or "").strip()
            if not t:
                return []
            parts = re.split(r"(?<=[.!?])\s+", t)
            return [p.strip() for p in parts if p.strip()]

        def limit_sentence_words(sent: str) -> str:
            ws = sent.split()
            if len(ws) <= max_words_sentence:
                # ensure sentence ends with punctuation
                return sent if sent.endswith(('.', '!', '?')) else sent.rstrip() + '.'
            trimmed = " ".join(ws[:max_words_sentence]).rstrip()
            # do NOT add ellipsis; finish cleanly with a period
            if not trimmed.endswith(('.', '!', '?')):
                trimmed += '.'
            return trimmed

        def synthesize_expl(r: Dict[str, Any]) -> str:
            skills = r.get("skills") or []
            exp = r.get("experience")
            core = ", ".join(skills[:5]) if skills else "relevant tech stack"
            exp_part = f"; {exp}" if isinstance(exp, str) and exp else ""
            return f"Core strengths: {core}{exp_part}."

        def _extract_name(r: Dict[str, Any]) -> str:
            # Prefer explicit name field if present
            name = r.get("name")
            if isinstance(name, str) and len(name.strip()) >= 2:
                return name.strip()
            txt_sources = [r.get("cv_text"), r.get("resume_text"), r.get("explanation")]
            banned = {"candidate", "resume", "description", "job", "profile"}
            # Pattern: 'Candidate James Baldwin' -> capture 'James Baldwin'
            pat_after_label = re.compile(r"\b(?:Candidate|CANDIDATE)\s+([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){1,2})\b")
            # Generic two-capitalized-words scanner
            pat_two_words = re.compile(r"\b([A-Z][a-zA-Z'-]+\s+[A-Z][a-zA-Z'-]+)\b")
            for txt in txt_sources:
                if not isinstance(txt, str) or len(txt) < 3:
                    continue
                head = txt[:400]
                m1 = pat_after_label.search(head)
                if m1:
                    cand = m1.group(1).strip()
                    if not any(w.lower() in banned for w in cand.split()):
                        return cand
                # Scan all candidates, choose first that passes filters
                for m in pat_two_words.finditer(head):
                    cand = m.group(1).strip()
                    if any(w.lower() in banned for w in cand.split()):
                        continue
                    # Skip patterns like 'Job Description'
                    if cand.lower() in {"job description", "curriculum vitae"}:
                        continue
                    return cand
            return "Candidate"

        lines: List[str] = []
        total_chars = 0
        for i, r in enumerate(results, start=1):
            cid = str(r.get("candidate_id") or r.get("id") or f"candidate-{i}")
            skills = r.get("skills") or []
            raw_expl = (r.get("explanation") or "").strip()
            if raw_expl:
                raw_expl = clean_generic(raw_expl, skills)
            sentences = split_sentences(raw_expl)

            if not sentences:
                sentences = split_sentences(synthesize_expl(r))

            # Ensure first sentence highlights skills explicitly if missing
            if sentences:
                if not any(sk.lower() in sentences[0].lower() for sk in skills[:3]):
                    if skills:
                        lead_sk = ", ".join(skills[:3])
                        sentences[0] = f"Possesses {lead_sk}. " + sentences[0]

            # Add achievement oriented snippet if not present and we have explanation tokens with action verbs
            action_verbs = ["built", "led", "designed", "developed", "implemented", "optimized", "improved", "deployed"]
            has_action = any(any(v in s.lower() for v in action_verbs) for s in sentences)
            if not has_action and raw_expl:
                # Extract phrase with an action verb from explanation (fallback to skill impact)
                match_action = re.search(r"(\b(?:built|led|designed|developed|implemented|optimized|improved|deployed)\b[^.]{10,120})", raw_expl, flags=re.IGNORECASE)
                if match_action:
                    sentences.append(match_action.group(1).strip().rstrip(".,") + ".")
                else:
                    if skills:
                        sentences.append(f"Applied {skills[0]} to deliver measurable outcomes.")

            # Truncate sentences count & words
            trimmed = [limit_sentence_words(s) for s in sentences[:max_sentences]]
            # Ensure 2–3 sentences requirement
            if len(trimmed) < 2 and skills:
                trimmed.append(f"Additional strengths: {', '.join(skills[:5])}." )

            # Compose block
            score = r.get("score")
            score_txt = ""
            try:
                if score is not None:
                    score_txt = f" (score {float(score):.2f})"
            except Exception:
                pass

            block = " ".join(trimmed)
            # Consistency: reuse cached phrasing if already seen for candidate
            if cid in self._candidate_cache:
                block = self._candidate_cache[cid]
            else:
                self._candidate_cache[cid] = block

            # Extract candidate display name (no id shown per requirement)
            display_name = _extract_name(r)
            # Final formatted line: numbering + name first, no internal id
            line = f"{i}. {display_name} — {block}{score_txt}".strip()
            total_chars += len(line)
            if total_chars > max_total_chars:
                break
            lines.append(line)

        return "\n".join(lines)

    def handle_message(self, message: str, forced_top_k: Optional[int] = None) -> str:
        msg = (message or "").strip()
        if not msg:
            return "Hi, I'm your AI recruiter. What role or skills are you looking for?"

        # Instant FAQ answers
        faq = self._faq_reply(msg)
        if faq:
            return faq

        intent = self._detect_intent(msg)

        if intent == "search_candidates":
            k = forced_top_k or self._infer_top_k(msg, self.chat_top_k)
            try:
                results = run_agent(job_desc=msg, top_k=k, with_explain=True)
            except TypeError:
                try:
                    results = run_agent(msg, top_k=k, with_explain=True)
                except TypeError:
                    results = run_agent(msg)
            # Enforce top_k strictly
            results = (results or [])[: max(int(k), 0)]
            return self._format_candidates(results)

        # --- new: calendar + email flow ---
        if intent == "schedule_interviews":
            return run_agent_schedule()

        # --- reuse last saved state and only send emails ---
        if intent == "email_candidates":
            return run_agent_email()

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
        results = (results or [])[: max(int(top_k), 0)]
        return self._format_candidates(results)

# Helper exposed for API to parse requested top_k directly
def parse_requested_top_k(text: str) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    t = re.sub(r"\b\d+\s*\+?\s*(years?|yrs?|months?)\b", "", t)
    # Try explicit patterns
    patterns = [
        r"top\s+(\d+)",
        r"top\s+(\d+)\s+(?:candidate|candidates|profiles)",
        r"(?:send|email|retrieve|get|show)\s+(?:the\s+)?(?:top\s+)?(\d+)\s+(?:candidate|candidates|profiles)",
        r"(\d+)\s+(?:best|top)\s+(?:candidate|candidates|profiles)",
        r"(?:candidate|candidates|profiles)\s*:\s*(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            try:
                k = int(m.group(1))
                return min(max(k, 1), 50)
            except Exception:
                pass
    # Word numbers
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "couple": 2, "few": 3, "single": 1
    }
    for w, val in number_words.items():
        if re.search(rf"\b{w}\b\s+(?:candidate|candidates|profiles)\b", t) or re.search(rf"\b(top|best)\s+{w}\b", t):
            return min(max(val, 1), 50)
    # Generic
    m2 = re.search(r"\b(\d+)\s+(?:candidate|candidates|profiles)\b", t)
    if m2:
        try:
            k = int(m2.group(1))
            return min(max(k, 1), 50)
        except Exception:
            pass
    # "best candidate"
    if re.search(r"\bbest\s+candidate\b", t):
        return 1
    return None
