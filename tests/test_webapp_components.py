"""Unit tests for webapp.components — unified confidence palette, SVG overlay, conf view."""
from webapp import components as C


def test_conf_color_buckets():
    assert C.conf_color(None, 0.5, 0.8) is None
    assert C.conf_color(0.1, 0.5, 0.8) == C.CONF_LOW
    assert C.conf_color(0.6, 0.5, 0.8) == C.CONF_MID
    assert C.conf_color(0.95, 0.5, 0.8) == C.CONF_HIGH


def test_overlay_region_mode_colors_labels_and_tables():
    blocks = [{"bbox": [0, 0, 10, 10], "label": "Text"}]
    tabs = [{"bbox": [5, 5, 20, 20], "label": "Table"}]
    svg = C.overlay_svg(blocks, tabs, "Region type")
    assert svg.count("<rect") == 2
    assert C.LABEL_COLORS["Text"] in svg
    assert C.LABEL_COLORS["Table"] in svg


def test_overlay_confidence_mode_uses_conf_palette():
    blocks = [{"bbox": [0, 0, 10, 10], "confidence": 0.05}]
    svg = C.overlay_svg(blocks, [], "Confidence")
    assert C.CONF_LOW in svg


def test_overlay_skips_malformed_bbox():
    assert C.overlay_svg([{"bbox": [1, 2]}], [], "Region type") == ""


def test_conf_view_html_tints_low_cells_and_escapes():
    grid = [["<b>", "ok"]]
    conf = [[0.01, 0.99]]
    out = C.conf_view_html(grid, conf)
    assert "&lt;b&gt;" in out          # escaped
    assert "background:#fde2e2" in out  # low cell tinted


def test_table_bbox_index_aligns_surya_tables_to_export_ids():
    surya_tables = [
        {"bbox": [0, 0, 50, 60]},
        {"bbox": []},  # no region bbox → skipped
    ]
    page_blocks = [{"table_id": "p1t1"}, {"table_id": "p1t2"}]
    idx = C.table_bbox_index(surya_tables, page_blocks)
    assert idx == {"p1t1": [0, 0, 50, 60]}


def test_highlight_rect_uses_bbox():
    svg = C.highlight_rect([10, 20, 40, 50])
    assert 'x="10.0"' in svg and 'width="30.0"' in svg and "#2563eb" in svg
