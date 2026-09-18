"""The data pipeline, including the value range the generator has to reach."""

import torch

import glyphgan as gg


def test_loader_range(dataset):
    """Targets must land inside tanh's reachable range, not on its asymptotes.

    The generator ends in Tanh, which approaches +/-1 without arriving. A target
    of exactly 1.0 needs infinite pre-activation and returns no gradient, and
    scraped glyphs are almost entirely pure black and pure white.
    """
    loader = gg.make_loader(dataset, size=16, batch_size=4, num_workers=0)
    x, _ = next(iter(loader))
    assert x.min() >= -gg.TARGET_SCALE - 1e-6, x.min()
    assert x.max() <= gg.TARGET_SCALE + 1e-6, x.max()
    # a fully black and fully white pixel must reach the ends, not fall short
    assert abs(x.min() + gg.TARGET_SCALE) < 1e-5
    assert abs(x.max() - gg.TARGET_SCALE) < 1e-5


def test_loader_yields_pairs(dataset):
    """ImageFolder yields (image, label); training must unpack, not iterate raw.

    Deleting the custom dataset class that silently dropped the label broke the
    training loop with `'list' object has no attribute 'to'`.
    """
    loader = gg.make_loader(dataset, size=16, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, type(batch)
    assert isinstance(batch[0], torch.Tensor)
