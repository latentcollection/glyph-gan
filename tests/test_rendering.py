"""Interpolation and the conversion to image frames."""

import pytest
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


def test_rendering_uses_the_generators_own_latent(tmp_path):
    """A generator with a non-default latent must not be fed module-default z."""
    gen = gg.Generator(16, latent=64, channels=32)
    cpu = torch.device("cpu")
    gg.save_preview(tmp_path / "p.png", gen, device=cpu)
    gg.render_interpolation(gen, tmp_path / "v.mp4", keys=2, frames=2, device=cpu)
    d, _ = gg.find_direction(gen, gg.measure_ink, n=64, batch=32, device=cpu)
    assert d.numel() == 64, d.numel()
    gg.render_direction(gen, d, tmp_path / "w.mp4", frames=3, device=cpu)


def test_rendering_restores_training_mode(tmp_path):
    """Rendering mid-run must not leave the generator frozen in eval().

    BatchNorm would stop updating for the rest of training, silently.
    """
    gen = gg.Generator(16, channels=32)
    gen.train()
    cpu = torch.device("cpu")
    gg.render_interpolation(gen, tmp_path / "v.mp4", keys=2, frames=2, device=cpu)
    assert gen.training, "render_interpolation left the generator in eval()"
    gg.find_direction(gen, gg.measure_ink, n=64, batch=32, device=cpu)
    assert gen.training, "find_direction left the generator in eval()"


def test_degenerate_frame_counts_are_rejected(tmp_path):
    gen = gg.Generator(16, channels=32)
    cpu = torch.device("cpu")
    d = torch.randn(gg.LATENT)
    with pytest.raises(ValueError):
        gg.render_direction(gen, d, tmp_path / "a.mp4", frames=1, device=cpu)
    with pytest.raises(ValueError):
        gg.render_interpolation(gen, tmp_path / "b.mp4", keys=2, frames=0, device=cpu)
