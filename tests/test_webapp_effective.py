"""What the run's 'Auto' choices actually resolved to (webapp/effective.py)."""
from types import SimpleNamespace

from webapp.effective import effective_dpi, effective_engine


def test_router_fallback_note_names_the_engine_that_ran():
    ws = ["something else", "[AutoRouter] fallback surya_kiri->surya | frac=0.310 cutoff=0.250"]
    assert effective_engine("auto", ws) == "surya"


def test_router_kept_note_names_the_engine_that_ran():
    ws = ["[AutoRouter] kept surya_kiri | frac=0.010 cutoff=0.250"]
    assert effective_engine("auto", ws) == "surya_kiri"


def test_auto_without_a_router_note_is_unknown_not_a_guess():
    """Before the OCR stage lands there is no decision to report — say so."""
    assert effective_engine("auto", []) is None


def test_explicit_engine_resolves_to_itself():
    assert effective_engine("surya", []) == "surya"


def test_no_request_at_all_is_unknown():
    assert effective_engine(None, []) is None


def test_effective_dpi_reads_the_resolved_render_dpi():
    doc = SimpleNamespace(ingest_result=SimpleNamespace(dpi=300))
    assert effective_dpi(doc) == 300


def test_effective_dpi_is_none_before_ingest():
    assert effective_dpi(SimpleNamespace(ingest_result=None)) is None
    assert effective_dpi(SimpleNamespace(ingest_result=SimpleNamespace())) is None
