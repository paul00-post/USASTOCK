"""
Gmail SMTP를 이용한 일일 리포트 발송.

필요한 .env 값:
  GMAIL_ADDRESS       — 보내는 Gmail 주소
  GMAIL_APP_PASSWORD  — Google 계정 2단계 인증 후 발급받은 16자리 앱 비밀번호
  GMAIL_TO            — 받는 이메일 주소 (보내는 주소와 같아도 됨)
"""

from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText

from utils.logger import get_logger

logger = get_logger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def send_report_email(subject: str, body: str) -> bool:
    address      = os.getenv("GMAIL_ADDRESS")
    app_password = os.getenv("GMAIL_APP_PASSWORD")
    to_address   = os.getenv("GMAIL_TO")

    if not address or not app_password or not to_address:
        logger.warning("GMAIL_ADDRESS/GMAIL_APP_PASSWORD/GMAIL_TO 미설정 — 이메일 발송 생략")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = address
    msg["To"]      = to_address

    try:
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(address, app_password)
            server.sendmail(address, [to_address], msg.as_string())
        logger.info("리포트 이메일 발송 완료: %s", subject)
        return True
    except Exception as e:
        logger.error("리포트 이메일 발송 실패: %s", e)
        return False
