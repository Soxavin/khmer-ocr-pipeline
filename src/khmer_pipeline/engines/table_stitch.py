from __future__ import annotations

# Geometric table de-fragmentation. Surya's layout model can shatter one dense
# table into many adjacent regions (real GDDE page 2 → a 2 row-band x 4 col-group
# grid of 8 Table boxes). Recognition then OCRs each region separately and
# serializes the content column-wise, destroying row<->value associations. We
# merge fragments that tile a contiguous area back into one master box *before*
# recognition runs. Pure geometry on (x0, y0, x1, y1) pixel tuples — no Surya import.
#
# Thresholds are in pixels, tuned for ~200 DPI renders (observed inter-fragment
# gaps were ~30-70 px). Two boxes are bridged when their gap is <= 2x the pad on
# that axis; keep these well below the spacing between genuinely separate tables.
_TABLE_MERGE_PAD_X = 60
_TABLE_MERGE_PAD_Y = 80
# Row-band variant: bridge fragments sharing a Y-band into a full-width strip. Pad
# is intentionally small (< typical inter-band gaps) so distinct bands stay separate.
_ROWBAND_PAD_Y = 25

_Box = tuple[float, float, float, float]


def _inflate(b: _Box, px: float, py: float) -> _Box:
    return (b[0] - px, b[1] - py, b[2] + px, b[3] + py)


def _intersects(a: _Box, b: _Box) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union(a: _Box, b: _Box) -> _Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def merge_table_regions(
    boxes: list[_Box],
    pad_x: float = _TABLE_MERGE_PAD_X,
    pad_y: float = _TABLE_MERGE_PAD_Y,
) -> list[_Box]:
    # Connected-components clustering: two boxes connect if, once inflated by the
    # pad, they intersect (handles both horizontal and vertical fragmentation).
    # Each component is unioned into one master box. Spatially separate tables
    # (gap > bridge distance) stay distinct.
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    inflated = [_inflate(b, pad_x, pad_y) for b in boxes]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _intersects(inflated[i], inflated[j]):
                parent[find(i)] = find(j)

    comps: dict[int, _Box] = {}
    for i in range(n):
        r = find(i)
        comps[r] = _union(comps[r], boxes[i]) if r in comps else boxes[i]

    merged = list(comps.values())
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def merge_table_rowbands(boxes: list[_Box], pad_y: float = _ROWBAND_PAD_Y) -> list[_Box]:
    # Cluster fragments by Y-band only (X ignored), union each band into a
    # full-width row strip. Keeps whole rows intact at a scale the VLM can read,
    # without collapsing the whole table into one oversized crop.
    n = len(boxes)
    if n <= 1:
        return list(boxes)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ai0, ai1 = boxes[i][1] - pad_y, boxes[i][3] + pad_y
        for j in range(i + 1, n):
            bj0, bj1 = boxes[j][1], boxes[j][3]
            if not (ai1 < bj0 or bj1 < ai0):  # Y-intervals overlap (i inflated)
                parent[find(i)] = find(j)

    comps: dict[int, _Box] = {}
    for i in range(n):
        r = find(i)
        comps[r] = _union(comps[r], boxes[i]) if r in comps else boxes[i]

    merged = list(comps.values())
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged
