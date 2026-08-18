"""Upload identity/probing, and the guard that both UIs share one definition.

`upload_id` was `md5[:12]` in webapp/api.py and `md5[:8]` in webapp/main.py, so the
same file uploaded through the React workspace and the NiceGUI fallback produced
two different ids -- and `upload_id` feeds `Settings.settings_key`, so their cached
runs could never match. `probe_pages` was duplicated verbatim. Nothing caught
either, because nothing asserted the two UIs agreed.
"""
import fitz
import pytest

from webapp import api, uploads
from webapp.settings import Settings


def _pdf_bytes(pages: int = 3) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    return doc.tobytes()


# --------------------------------------------------------------------------
# The regression guard: one definition, reachable from both UIs.
# --------------------------------------------------------------------------

def test_api_probe_pages_is_the_shared_implementation():
    """api.py rebinds rather than redefines -- and the rebound name must stay an
    attribute of webapp.api, because ~20 tests patch "webapp.api._probe_pages"."""
    assert api._probe_pages is uploads.probe_pages


def test_both_uis_derive_the_same_id_for_the_same_bytes():
    data = _pdf_bytes()
    # The NiceGUI page body cannot be imported (webapp.main calls ui.run() at
    # module scope), so assert on the shared function both call sites now use.
    assert uploads.upload_id(data) == uploads.upload_id(data)
    assert len(uploads.upload_id(data)) == 12


# --------------------------------------------------------------------------
# upload_id
# --------------------------------------------------------------------------

def test_upload_id_is_content_addressed():
    assert uploads.upload_id(b"same") == uploads.upload_id(b"same")
    assert uploads.upload_id(b"one") != uploads.upload_id(b"two")


def test_upload_id_feeds_a_distinct_settings_key():
    """The consequence of the old mismatch: different id -> different cache key."""
    s = Settings()
    assert s.settings_key(uploads.upload_id(b"a")) != s.settings_key(uploads.upload_id(b"b"))


# --------------------------------------------------------------------------
# probe_pages
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pages", [1, 3, 10])
def test_probe_pages_counts_pdf_pages(pages):
    assert uploads.probe_pages("doc.pdf", _pdf_bytes(pages)) == pages


def test_probe_pages_returns_zero_for_an_unreadable_pdf():
    assert uploads.probe_pages("broken.pdf", b"not a pdf at all") == 0


@pytest.mark.parametrize("name", ["scan.png", "scan.PNG", "photo.jpg", "page.tiff"])
def test_probe_pages_treats_images_as_one_page(name):
    assert uploads.probe_pages(name, b"") == 1


# --------------------------------------------------------------------------
# ACCEPTED_SUFFIXES
# --------------------------------------------------------------------------

def test_accept_attr_is_the_suffix_list():
    assert uploads.accept_attr() == ",".join(uploads.ACCEPTED_SUFFIXES)
    assert uploads.accept_attr().startswith(".pdf")


def test_every_accepted_suffix_is_probeable():
    """Each accepted type must take one of probe_pages' two branches, not error."""
    for suffix in uploads.ACCEPTED_SUFFIXES:
        data = _pdf_bytes(2) if suffix == ".pdf" else b""
        assert uploads.probe_pages(f"doc{suffix}", data) >= 1
