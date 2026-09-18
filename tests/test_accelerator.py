"""The real device.

CPU-only tests hide backend gaps: MPS has no grid_sample border padding, no
grid_sampler backward and no foreach kernels, and all three slipped through a
green CPU run."""

import pytest
import torch

import glyphgan as gg


@pytest.mark.accelerator
def test_accelerator():
    """Exercise the real device.

    CPU-only tests hide backend gaps - MPS has no grid_sample border padding
    and no foreach optimiser kernels, and both slipped through a green CPU run.
    """
    dev = gg.pick_device()
    if dev.type == "cpu":
        print("    (no accelerator; skipped)")
        return
    gen, dis = gg.Generator(64, channels=64).to(dev), gg.Discriminator(64, channels=64).to(dev)
    og = torch.optim.Adam(gen.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(dis.parameters(), 2e-4, betas=(0.5, 0.999))
    x = torch.rand(4, 1, 64, 64) * 1.8 - 0.9
    ld, lg = gg.train_step(gen, dis, og, od, x, dev, augment=True)
    assert ld == ld and lg == lg, "loss is NaN on accelerator"
    gg.EMA(gen).update(gen)
    out = gen(torch.randn(2, gg.LATENT, device=dev))
    assert out.shape == (2, 1, 64, 64)
