from __future__ import annotations
import numpy as np
from khmer_pipeline.generate_degraded import degrade_page


def _img() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)


def test_shape_and_dtype_preserved():
    img = _img()
    out = degrade_page(img, seed=1)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_deterministic_for_fixed_seed():
    img = _img()
    a = degrade_page(img, seed=7)
    b = degrade_page(img, seed=7)
    assert np.array_equal(a, b)


def test_actually_changes_pixels():
    img = _img()
    out = degrade_page(img, seed=3)
    assert not np.array_equal(out, img)


def test_different_seed_differs():
    img = _img()
    a = degrade_page(img, seed=1)
    b = degrade_page(img, seed=2)
    assert not np.array_equal(a, b)
