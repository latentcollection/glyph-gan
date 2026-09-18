"""Measured proxies and the latent directions derived from them."""

import torch

import glyphgan as gg


def test_measures_on_known_shapes():
    """measure_ink and measure_extent are proxies with defined meanings."""
    blank = torch.full((1, 1, 16, 16), -gg.TARGET_SCALE)
    assert gg.measure_ink(blank).item() == 0.0

    half = blank.clone()
    half[:, :, :8, :] = gg.TARGET_SCALE
    assert abs(gg.measure_ink(half).item() - 0.5) < 1e-6

    wide = blank.clone()
    wide[:, :, 6:10, 2:14] = gg.TARGET_SCALE  # 12 wide, 4 tall
    assert abs(gg.measure_extent(wide).item() - 3.0) < 1e-6


def test_find_direction_recovers_a_planted_axis():
    """Plant a known axis in a fake generator and check it is recovered."""

    class Planted(torch.nn.Module):
        def forward(self, z):
            t = torch.tanh(z[:, :1]).view(-1, 1, 1, 1)
            return (t * gg.TARGET_SCALE).expand(-1, 1, 16, 16)

    d, corr = gg.find_direction(
        Planted(), gg.measure_ink, n=1024, batch=64, device=torch.device("cpu")
    )
    assert corr > 0.5, corr
    assert d.abs().argmax().item() == 0, d.abs().argmax().item()
