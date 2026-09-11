"""P4 hardening tests: email normalization, context budget, query limits."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.rag


class TestEmailNormalization:
    def test_register_lowercases_full_address(self):
        from app.schemas.auth import RegisterRequest

        r = RegisterRequest(email="User.Name@Example.COM", password="secret12")
        assert r.email == "user.name@example.com"

    def test_login_strips_and_lowercases(self):
        from app.schemas.auth import LoginRequest

        r = LoginRequest(email="  MiXeD@Case.io  ", password="x")
        assert r.email == "mixed@case.io"

    def test_forgot_password_normalizes(self):
        from app.schemas.auth import ForgotPasswordRequest

        r = ForgotPasswordRequest(email="HELP@Domain.ORG")
        assert r.email == "help@domain.org"

    def test_otp_verify_normalizes(self):
        from app.schemas.auth import OTPVerifyRequest

        r = OTPVerifyRequest(email="A@B.Com", otp_code="123456")
        assert r.email == "a@b.com"


class TestChatRequestLimits:
    def test_query_max_length_enforced(self):
        from pydantic import ValidationError

        from app.schemas.conversation import ChatRequest

        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 10_001)

    def test_query_within_limit_ok(self):
        from app.schemas.conversation import ChatRequest

        req = ChatRequest(query="x" * 10_000)
        assert len(req.query) == 10_000


