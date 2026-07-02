from __future__ import annotations
from unittest.mock import patch
import khmer_pipeline.utils.backend_status as bs


def test_running_true_when_pgrep_finds_pids():
    # pgrep exits 0 and prints PIDs when matches exist
    with patch("khmer_pipeline.utils.backend_status.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "12345\n12346\n"
        assert bs.llama_server_running() is True


def test_running_false_when_pgrep_no_match():
    # pgrep exits 1 with empty stdout when nothing matches
    with patch("khmer_pipeline.utils.backend_status.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stdout = ""
        assert bs.llama_server_running() is False


def test_running_false_when_pgrep_missing():
    # pgrep not on PATH -> treated as not running, never raises
    with patch("khmer_pipeline.utils.backend_status.subprocess.run", side_effect=FileNotFoundError):
        assert bs.llama_server_running() is False


def test_running_false_on_unexpected_error():
    with patch("khmer_pipeline.utils.backend_status.subprocess.run", side_effect=OSError("boom")):
        assert bs.llama_server_running() is False


def test_pids_parsed():
    with patch("khmer_pipeline.utils.backend_status.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = "12345\n12346\n\n"
        assert bs.llama_server_pids() == [12345, 12346]


def test_pids_empty_when_none():
    with patch("khmer_pipeline.utils.backend_status.subprocess.run") as run:
        run.return_value.returncode = 1
        run.return_value.stdout = ""
        assert bs.llama_server_pids() == []
