import logging

from app.tasks.celery_app import celery_app

log = logging.getLogger(__name__)


@celery_app.task(bind=True, name="tasks.send_verification_email",
                 max_retries=3, default_retry_delay=60, retry_backoff=True,
                 retry_jitter=True, queue="email")
def send_verification_email(self, to: str, otp: str, token: str) -> None:
    try:
        from app.services.email_service import email_service
        email_service.send_verification(to, otp, token)
    except Exception as exc:
        log.error("Verification email failed", extra={"to": to, "error": str(exc)})
        raise self.retry(exc=exc) from exc


@celery_app.task(bind=True, name="tasks.send_password_reset_email",
                 max_retries=3, default_retry_delay=60, retry_backoff=True,
                 retry_jitter=True, queue="email")
def send_password_reset_email(self, to: str, otp: str, token: str) -> None:
    try:
        from app.services.email_service import email_service
        email_service.send_password_reset(to, otp, token)
    except Exception as exc:
        log.error("Reset email failed", extra={"to": to, "error": str(exc)})
        raise self.retry(exc=exc) from exc
