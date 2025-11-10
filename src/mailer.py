"""mailer.py

Utility for sending candidate invitation emails via SMTP.
Credentials and behavior are controlled by environment variables so we never
commit secrets to the repository.

Environment variables (set with Windows PowerShell `setx VAR "value"` then
restart shell):
  SMTP_HOST      e.g. smtp.gmail.com or smtp.office365.com
  SMTP_PORT      default 587 (STARTTLS)
  SMTP_USER      full email address of sender account
  SMTP_PASS      app password / SMTP password (NEVER commit plain password!)
  FROM_EMAIL     optional override for From header (defaults to SMTP_USER)
  EMAIL_DRY_RUN  "1" (default) to log instead of sending; set to "0" to send

NOTE: For Gmail personal accounts you should enable 2FA and create an App
Password; the raw account password will usually be blocked by modern security.
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

def _env(key: str, default: str | None = None) -> str:
    val = os.getenv(key)
    return val if val is not None else (default or "")

def _load_cfg():
    host = _env("SMTP_HOST", "smtp.gmail.com")
    port_raw = _env("SMTP_PORT", "587")
    try:
        port = int(port_raw)
    except Exception:
        port = 587
    user = _env("SMTP_USER", "")
    pwd = _env("SMTP_PASS", "")
    from_email = _env("FROM_EMAIL", user)
    dry = _env("EMAIL_DRY_RUN", "1")
    return host, port, user, pwd, from_email, dry


def send_email(to_email: str, subject: str, body: str, reply_to: Optional[str] = None) -> bool:
    """Send an email (or log if dry-run).

    Returns True if send (or dry-run) succeeded, False if skipped.
    """
    if not to_email or to_email == "unknown":
        return False
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FROM_EMAIL, EMAIL_DRY_RUN = _load_cfg()
    if EMAIL_DRY_RUN == "1":
        print(f"[DRY RUN] To: {to_email}\nFrom: {FROM_EMAIL}\nSubject: {subject}\nBody (truncated): {body[:300]}...")
        return True

    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    if reply_to:
        msg["Reply-To"] = reply_to
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=45) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, [to_email], msg.as_string())
        print(f"[EMAIL SENT] {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {to_email}: {e}")
        return False


def render_invite_subject(role: Optional[str] = None) -> str:
    return f"Interview Invitation{f' for {role}' if role else ''}"


def render_invite_body(candidate_name: Optional[str], job_desc_snippet: str, reply_to: Optional[str] = None) -> str:
    name = candidate_name or "Candidate"
    footer = f"\n\nPlease reply to {reply_to}" if reply_to else ""
    return (
        f"Hello {name},\n\n"
        f"Thank you for your application. We'd like to invite you to an interview.\n\n"
        f"Role details (snippet): {job_desc_snippet[:500]}\n"
        f"\nBest regards,\nRecruitment Team"
        f"{footer}"
    )


__all__ = [
    "send_email",
    "render_invite_subject",
    "render_invite_body",
]
