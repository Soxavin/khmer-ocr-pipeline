"""Per-client state for the review UIs: staleness, run reset, and the document list.

`apps/api/state.py` had no direct test despite owning three things that silently
corrupt a session when wrong: whether cached results are still valid for the
current settings, whether Retry actually clears the previous run, and whether the
active-document index stays in range as documents come and go.
"""
from apps.api.settings import Settings
from apps.api.state import AppState, Document, Progress


def _doc(upload_id: str = "abc123", name: str = "doc.pdf") -> Document:
    return Document(upload_name=name, upload_bytes=b"x", upload_id=upload_id, doc_page_count=3)


def _ran(doc: Document, settings: Settings) -> Document:
    """Mark a document as having completed a run under `settings`."""
    doc.export_result = object()
    doc.last_key = settings.settings_key(doc.upload_id)
    return doc


# --------------------------------------------------------------------------
# results_are_stale
# --------------------------------------------------------------------------

def test_fresh_results_are_not_stale():
    s = Settings()
    assert _ran(_doc(), s).results_are_stale(s) is False


def test_a_pipeline_affecting_change_makes_results_stale():
    s = Settings()
    doc = _ran(_doc(), s)
    s.dpi = 300
    assert doc.results_are_stale(s) is True


def test_a_display_only_change_does_not_make_results_stale():
    """overlay_mode/show_layout only affect rendering, so they must not force a re-run."""
    s = Settings()
    doc = _ran(_doc(), s)
    s.overlay_mode = "Confidence"
    s.show_layout = not s.show_layout
    assert doc.results_are_stale(s) is False


def test_a_document_with_no_results_is_never_stale():
    s = Settings()
    doc = _doc()
    s.dpi = 300
    assert doc.results_are_stale(s) is False


def test_staleness_is_per_document_id():
    """settings_key is salted with upload_id, so one document's key cannot validate another's."""
    s = Settings()
    doc = _ran(_doc("aaa"), s)
    other = Document(upload_name="b.pdf", upload_bytes=b"y", upload_id="bbb")
    other.export_result = object()
    other.last_key = doc.last_key
    assert other.results_are_stale(s) is True


# --------------------------------------------------------------------------
# reset_run
# --------------------------------------------------------------------------

def test_reset_run_clears_results_and_edits_but_keeps_the_upload():
    s = Settings()
    doc = _ran(_doc(), s)
    doc.run_error = "boom"
    doc.edited_tables = {"t1": [["a"]]}
    doc.edited_text = {0: "edited"}
    doc.reviewed = {"t1": True}
    doc.captured = {"t1"}
    doc.current_page_idx = 4
    doc.selected = ("t1", 0, 0)
    doc.run_page_indices = [0, 1]

    doc.reset_run()

    assert doc.upload_name == "doc.pdf" and doc.upload_bytes == b"x"
    assert doc.upload_id == "abc123" and doc.doc_page_count == 3
    assert doc.has_results is False
    assert (doc.run_error, doc.last_key, doc.run_page_indices) == (None, None, None)
    assert doc.edited_tables == {} and doc.edited_text == {}
    assert doc.reviewed == {} and doc.captured == set()
    assert doc.current_page_idx == 0 and doc.selected is None


def test_reset_run_installs_a_fresh_progress():
    """A carried-over cancel_requested would abort the very next run at stage one."""
    doc = _doc()
    doc.progress = Progress(active=True, cancel_requested=True, page=2, fraction=0.5)
    doc.reset_run()
    assert doc.progress.cancel_requested is False and doc.progress.active is False
    assert (doc.progress.page, doc.progress.fraction) == (0, 0.0)


def test_reset_run_keeps_the_preprocess_suggestion():
    """It derives from the upload bytes, not the run, so re-deriving it would be waste."""
    doc = _doc()
    doc.preprocess_suggestion = {"suggested": {"deskew": True}}
    doc.reset_run()
    assert doc.preprocess_suggestion == {"suggested": {"deskew": True}}


# --------------------------------------------------------------------------
# AppState
# --------------------------------------------------------------------------

def test_add_document_appends_and_activates():
    st = AppState()
    st.add_document("a.pdf", b"a", "id-a", 1)
    st.add_document("b.pdf", b"b", "id-b", 2)
    assert [d.upload_name for d in st.documents] == ["a.pdf", "b.pdf"]
    assert st.doc().upload_id == "id-b"


def test_re_adding_the_same_content_selects_it_instead_of_duplicating():
    st = AppState()
    st.add_document("a.pdf", b"a", "id-a", 1)
    st.add_document("b.pdf", b"b", "id-b", 2)
    st.add_document("a-again.pdf", b"a", "id-a", 1)
    assert len(st.documents) == 2
    assert st.doc().upload_id == "id-a"


def test_doc_clamps_an_out_of_range_active_index():
    st = AppState()
    st.add_document("a.pdf", b"a", "id-a", 1)
    st.active = 99
    assert st.doc().upload_id == "id-a" and st.active == 0


def test_doc_is_none_when_empty_and_after_clearing():
    st = AppState()
    assert st.doc() is None
    st.add_document("a.pdf", b"a", "id-a", 1)
    st.clear_documents()
    assert st.doc() is None and st.documents == [] and st.active == 0


def test_each_appstate_gets_its_own_settings_and_documents():
    """AppState is per browser connection; shared mutable defaults would leak across clients."""
    a, b = AppState(), AppState()
    a.settings.dpi = 300
    a.add_document("a.pdf", b"a", "id-a", 1)
    assert b.settings.dpi == "auto" and b.documents == []
