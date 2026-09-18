"""Interpolation and the conversion to image frames."""

import torch

import glyphgan as gg


def test_slerp_norm():
    a, b = torch.randn(1, 200), torch.randn(1, 200)
    mid = gg.slerp(a, b, 0.5).norm().item()
    ends = (a.norm().item() + b.norm().item()) / 2
    # A linear midpoint collapses toward zero; a spherical one should not.
    assert mid > 0.9 * ends, f"slerp midpoint norm {mid:.2f} vs ends {ends:.2f}"


def test_to_image_maps_extremes():
    lo = gg.to_image(torch.full((1, 4, 4), -gg.TARGET_SCALE))
    hi = gg.to_image(torch.full((1, 4, 4), gg.TARGET_SCALE))
    assert lo.min() == lo.max() == 0, lo.min()
    assert hi.min() == hi.max() == 255, hi.max()
