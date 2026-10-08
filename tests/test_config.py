"""`CORS_ORIGINS` must accept every documented form without crashing `Settings()` at startup.

pydantic-settings JSON-decodes list fields before validators run unless the field is marked `NoDecode`;
without it the `.env.example` value `*` and the README's comma-separated form both raised `SettingsError`,
so `cp .env.example .env` (the quick start) produced a server that would not start.
"""

from __future__ import annotations

import pytest

from smartup_rag.config import PROJECT_ROOT, Settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("*", ["*"]),
        (
            "https://www.smartupacademy.org,https://smartupacademy.org",
            ["https://www.smartupacademy.org", "https://smartupacademy.org"],
        ),
        (" https://a.example , https://b.example ", ["https://a.example", "https://b.example"]),
        ('["https://www.smartupacademy.org"]', ["https://www.smartupacademy.org"]),
    ],
)
def test_cors_origins_accepts_documented_forms(monkeypatch, raw: str, expected: list[str]) -> None:
    monkeypatch.setenv("CORS_ORIGINS", raw)
    assert Settings(_env_file=None).cors_origins == expected


def test_cors_origins_default(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert Settings(_env_file=None).cors_origins == ["*"]


def test_env_example_template_loads(monkeypatch) -> None:
    """The template the README tells users to copy must itself parse."""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    s = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert s.cors_origins == ["*"]
