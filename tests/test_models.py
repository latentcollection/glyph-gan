"""Architecture: shapes, capacity and the size contract."""

import torch

import glyphgan as gg


def test_shapes():
    for size in (64, 128):
        gen, dis = gg.Generator(size), gg.Discriminator(size)
        out = gen(torch.randn(2, gg.LATENT))
        assert out.shape == (2, 1, size, size), f"{size}: {out.shape}"
        assert dis(out).shape == (2, 1), f"{size}: {dis(out).shape}"
        assert out.min() >= -1.0 and out.max() <= 1.0, "tanh range"


def test_channels():
    small = gg.Generator(64, channels=256)
    big = gg.Generator(64, channels=1024)
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in big.parameters())


def test_stages_rejects_bad_size():
    for bad in (5, 6, 100, 2):
        try:
            gg._stages(bad)
        except ValueError:
            continue
        raise AssertionError(f"size {bad} should have been rejected")
