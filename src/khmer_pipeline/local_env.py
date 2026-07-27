"""Load a local ``.env`` into the environment — dependency-free.

Desktop convenience: an API key (e.g. ``GEMINI_API_KEY``) can live in a
gitignored ``.env`` at the repo root instead of being re-exported every shell.
A real shell export always WINS — values are only filled in when the variable is
not already set — so ``.env`` is a fallback, never an override. No python-dotenv;
a tiny KEY=VALUE parser keeps the local-first dependency surface minimal.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_local_env(path: Path | None = None) -> int:
    """Populate os.environ from a .env file (repo root by default).

    Parses ``KEY=VALUE`` lines; ignores blanks and ``#`` comments; tolerates a
    leading ``export`` and surrounding quotes. Never overwrites an existing
    variable. Returns the number of keys set; a missing file is a no-op (0)."""
    env_path = path or (_REPO_ROOT / ".env")
    try:
        text = env_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return 0

    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):  # tolerate a pasted shell export line
            key = key[len("export "):].strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count
