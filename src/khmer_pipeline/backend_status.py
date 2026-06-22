from __future__ import annotations
import subprocess

# Surya 0.20 spawns a resident llama-server (KEEP_ALIVE=true) for OCR. We detect
# it by process name rather than a port, since Surya picks the port internally.
_MATCH = "llama-server"


def llama_server_pids() -> list[int]:
    try:
        proc = subprocess.run(
            ["pgrep", "-f", _MATCH],
            capture_output=True, text=True,
        )
    except (FileNotFoundError, OSError):
        return []
    if proc.returncode != 0:
        return []
    return [int(p) for p in proc.stdout.split() if p.strip().isdigit()]


def llama_server_running() -> bool:
    return bool(llama_server_pids())
