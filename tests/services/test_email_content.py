"""Email content tests (no SMTP, no task queue).

Sends go through ``EmailService`` directly in-process; ``_send`` is mocked
so these assert on the rendered subject + body instead of delivery.
"""
from __future__ import annotations


class TestSendVerification:
    def test_body_contains_otp_and_token_link(self, monkeypatch):
        from app.services.email_service import EmailService

        captured = {}

        def fake_send(self, to, subject, html):
            captured.update(to=to, subject=subject, html=html)

        monkeypatch.setattr(EmailService, "_send", fake_send)
        EmailService().send_verification("user@example.com", "123456", "tok-abc")

        assert captured["to"] == "user@example.com"
        assert "123456" in captured["html"]
        assert "/verify-email?token=tok-abc" in captured["html"]
        assert "Verify" in captured["subject"]


class TestSendPasswordReset:
    def test_body_contains_otp_and_token_link(self, monkeypatch):
        from app.services.email_service import EmailService

        captured = {}

        def fake_send(self, to, subject, html):
            captured.update(to=to, subject=subject, html=html)

        monkeypatch.setattr(EmailService, "_send", fake_send)
        EmailService().send_password_reset("user@example.com", "654321", "tok-xyz")

        assert captured["to"] == "user@example.com"
        assert "654321" in captured["html"]
        assert "/reset-password?token=tok-xyz" in captured["html"]
        assert "Reset" in captured["subject"]
