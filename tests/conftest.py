"""Fixtures shared across the suite."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dataset(tmp_path):
    """A tiny ImageFolder-shaped dataset of glyph-ish blobs."""
    size, n = 16, 8
    d = tmp_path / "data" / "a"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = np.zeros((size, size), dtype="uint8")
        img[4:12, 3 + (i % 3) : 9 + (i % 3)] = 255
        img = np.clip(img + rng.integers(0, 40, img.shape), 0, 255).astype("uint8")
        Image.fromarray(img).save(d / f"face{i}.png")
    return tmp_path / "data"
