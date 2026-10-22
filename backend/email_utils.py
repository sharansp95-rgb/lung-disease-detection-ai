"""
Email sending via Gmail SMTP (smtplib, stdlib -- no extra dependency).
Falls back to printing the email content (including the actual link) to
the console/log if GMAIL_ADDRESS/GMAIL_APP_PASSWORD aren't set, or if the
send itself fails for any reason -- verification/reset flows must stay
testable without real credentials, per spec.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import EMAIL_CONFIGURED, GMAIL_ADDRESS, GMAIL_APP_PASSWORD

logger = logging.getLogger("ct_xray_api.email")

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
# Sending happens inside the signup/forgot-password request, so a blocked or
# unreachable SMTP server must fail fast (to the console fallback) rather
# than hold the request. Applies to the connect and to each later step.
SMTP_TIMEOUT_SECONDS = 5

APP_NAME = "Lung Disease Detection"
ACCENT = "#3f6fee"


def _html_template(heading: str, intro: str, button_text: str, link: str, expiry_note: str) -> str:
    return f"""\
<div style="font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 480px; margin: 0 auto;">
  <div style="background:{ACCENT}; padding: 22px 32px; border-radius: 12px 12px 0 0;">
    <span style="color:#ffffff; font-size:19px; font-weight:800; letter-spacing:-0.01em;">{APP_NAME}</span>
  </div>
  <div style="background:#ffffff; padding: 32px; border-radius: 0 0 12px 12px; border: 1px solid #e4e4e7; border-top:none;">
    <h1 style="margin:0 0 12px; font-size: 20px; color:#18181b;">{heading}</h1>
    <p style="margin:0 0 4px; font-size: 14px; line-height:1.6; color:#52525b;">{intro}</p>
    <p style="margin:0 0 24px; font-size: 13px; line-height:1.6; color:#8a8a92;">{expiry_note}</p>
    <a href="{link}" style="display:inline-block; background:{ACCENT}; color:#ffffff; text-decoration:none;
       font-weight:600; font-size:14px; padding:12px 26px; border-radius:8px;">{button_text}</a>
    <p style="margin:28px 0 0; font-size:12px; color:#8a8a92;">Or copy this link into your browser:</p>
    <p style="margin:4px 0 0; font-size:12px; word-break:break-all;">
      <a href="{link}" style="color:{ACCENT};">{link}</a>
    </p>
  </div>
  <p style="text-align:center; font-size:11px; color:#a3a3a3; margin-top:16px;">
    This email was sent by {APP_NAME}. Please do not reply directly.
  </p>
</div>
"""


def _console_fallback(to_email: str, subject: str, text_body: str):
    logger.warning(
        f"\n{'='*60}\n"
        f"EMAIL NOT SENT (Gmail not configured or send failed) -- printing instead:\n"
        f"To: {to_email}\n"
        f"Subject: {subject}\n"
        f"{'-'*60}\n"
        f"{text_body}\n"
        f"{'='*60}"
    )


def send_email(to_email: str, subject: str, text_body: str, html_body: str) -> bool:
    """Returns True if actually emailed, False if it fell back to console
    (not an error -- expected/logged behavior when Gmail isn't configured).
    Sends multipart/alternative (plain text + HTML) so clients that can't
    render HTML still get a readable, working link."""
    if not EMAIL_CONFIGURED:
        _console_fallback(to_email, subject, text_body)
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = GMAIL_ADDRESS
        msg["To"] = to_email
        # Text part first, HTML second -- clients render the last part they
        # support, so HTML-capable clients show the styled version while
        # plain-text clients fall back to the text part.
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, [to_email], msg.as_string())
        logger.info(f"Email sent to {to_email}: {subject}")
        return True
    except Exception as e:
        logger.warning(f"Gmail send failed ({e}), falling back to console.")
        _console_fallback(to_email, subject, text_body)
        return False


def send_verification_email(to_email: str, link: str):
    text_body = (
        f"Welcome! Please verify your email by visiting this link (expires in 24 hours):\n\n{link}\n\n"
        f"If you didn't sign up for this, you can ignore this email."
    )
    html_body = _html_template(
        heading="Verify your email address",
        intro="Welcome! Click the button below to verify your account and start using the tool.",
        button_text="Verify Email",
        link=link,
        expiry_note="This link expires in 24 hours. If you didn't sign up for this, you can ignore this email.",
    )
    send_email(to_email, "Verify your email -- Lung Disease Detection", text_body, html_body)


def send_password_reset_email(to_email: str, link: str):
    text_body = (
        f"You requested a password reset. Visit this link to set a new password "
        f"(expires in 1 hour):\n\n{link}\n\n"
        f"If you didn't request this, you can safely ignore this email."
    )
    html_body = _html_template(
        heading="Reset your password",
        intro="You requested a password reset. Click the button below to choose a new password.",
        button_text="Reset Password",
        link=link,
        expiry_note="This link expires in 1 hour. If you didn't request this, you can safely ignore this email.",
    )
    send_email(to_email, "Password reset -- Lung Disease Detection", text_body, html_body)
