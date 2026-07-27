"""Tests for the dependency-free .env loader (local_env.py)."""
from __future__ import annotations

from khmer_pipeline.local_env import load_local_env


def test_loads_key_value_pairs(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY=abc123\n', encoding="utf-8")
    n = load_local_env(env)
    assert n == 1
    import os
    assert os.environ["GEMINI_API_KEY"] == "abc123"


def test_shell_export_wins_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    load_local_env(env)
    import os
    assert os.environ["GEMINI_API_KEY"] == "from-shell"  # never overwritten


def test_tolerates_export_prefix_and_quotes(monkeypatch, tmp_path):
    monkeypatch.delenv("K", raising=False)
    monkeypatch.delenv("Q", raising=False)
    env = tmp_path / ".env"
    env.write_text('export K=plain\nQ="quoted value"\n', encoding="utf-8")
    load_local_env(env)
    import os
    assert os.environ["K"] == "plain"
    assert os.environ["Q"] == "quoted value"


def test_ignores_comments_and_blanks(monkeypatch, tmp_path):
    monkeypatch.delenv("REAL", raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\n   \nREAL=1\n# GEMINI_MODEL=x\n", encoding="utf-8")
    assert load_local_env(env) == 1
    import os
    assert os.environ["REAL"] == "1"


def test_missing_file_is_a_noop(tmp_path):
    assert load_local_env(tmp_path / "does-not-exist") == 0
