"""Every Settings default must be selectable in the NiceGUI fallback's option maps.

NiceGUI's `ui.select` raises `ValueError: Invalid value: X` when the bound value is
absent from its options, and `@ui.page` turns that into a 500 for the whole page.
So a default drifting out of its own option set is a hard outage, not a cosmetic
mismatch — and it is invisible to the React workspace, which builds its picker from
`api._ENGINES` instead. `apps.api.main` cannot be imported here (it calls `ui.run()` at
module scope), which is why the option maps live in `settings.py`.
"""
import pytest

from apps.api.settings import (
    DPI_OPTIONS,
    ENGINE_OPTIONS,
    EXTRACTION_MODE_OPTIONS,
    OVERLAY_MODE_OPTIONS,
    SCOPE_OPTIONS,
    Settings,
    with_current,
)


@pytest.mark.parametrize("field, options", [
    ("dpi", DPI_OPTIONS),
    ("ocr_engine_key", ENGINE_OPTIONS),
    ("page_scope", SCOPE_OPTIONS),
    ("overlay_mode", OVERLAY_MODE_OPTIONS),
    ("tables_only", EXTRACTION_MODE_OPTIONS),
])
def test_every_default_is_selectable(field, options):
    assert getattr(Settings(), field) in options


def test_overlay_mode_domain_matches_the_renderer():
    """components.py branches on these exact strings; a rename here is a silent no-op there."""
    assert set(OVERLAY_MODE_OPTIONS) == {"Region type", "Confidence"}


def test_extraction_mode_covers_both_boolean_states():
    assert set(EXTRACTION_MODE_OPTIONS) == {False, True}


def test_engine_options_are_real_registry_engines():
    """A label pointing at a key the registry cannot resolve fails only at run time."""
    from khmer_pipeline.engines import engine_registry

    valid = set(engine_registry._OCR_ENGINES)
    assert set(ENGINE_OPTIONS) <= valid, f"unknown engines: {set(ENGINE_OPTIONS) - valid}"


def test_with_current_passes_through_a_known_value():
    assert with_current(SCOPE_OPTIONS, "all") is SCOPE_OPTIONS


def test_with_current_admits_a_value_this_ui_does_not_enumerate():
    # page_scope="list" and the Labs engines are reachable from the React workspace
    # but have no NiceGUI widget; they must not take the fallback page down.
    out = with_current(SCOPE_OPTIONS, "list")
    assert "list" in out and out["list"] == "list — set in the main workspace"
    assert SCOPE_OPTIONS == {"all": "All pages", "single": "Single page", "range": "Page range"}


def test_with_current_admits_a_labs_engine():
    assert "gemma_ardb" in with_current(ENGINE_OPTIONS, "gemma_ardb")


@pytest.mark.parametrize("dpi, expected", [("auto", 200), (150, 150), (300, 300)])
def test_dpi_estimate_is_always_numeric(dpi, expected):
    """The job-size heads-up multiplies by DPI; "auto" would raise TypeError."""
    s = Settings()
    s.dpi = dpi
    assert s.dpi_estimate == expected
    assert isinstance(s.dpi_estimate / 200.0, float)
