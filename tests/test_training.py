"""A training step, its label noise and augment, and the whole loop."""

import pytest
import torch

import glyphgan as gg


def test_train_step():
    dev = torch.device("cpu")
    gen, dis = gg.Generator(64).to(dev), gg.Discriminator(64).to(dev)
    og = torch.optim.Adam(gen.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(dis.parameters(), 2e-4, betas=(0.5, 0.999))
    x = torch.rand(2, 1, 64, 64) * 1.8 - 0.9
    ld, lg = gg.train_step(gen, dis, og, od, x, dev)
    assert ld == ld and lg == lg, "loss is NaN"


def test_labels_uniform():
    counts = torch.zeros(8)
    for _ in range(2000):
        real, _ = gg._soft_labels((8, 1), torch.device("cpu"), flip_rate=0.03)
        counts += (real.squeeze() < 0.5).float()
    rate = counts / 2000
    assert rate.max() < 0.08, f"flips not uniform across the batch: {rate.tolist()}"


def test_augment():
    x = (torch.rand(4, 1, 64, 64) * 1.8 - 0.9).requires_grad_(True)
    y = gg.diff_augment(x)
    assert y.shape == x.shape, y.shape
    y.sum().backward()
    assert x.grad.abs().sum() > 0, "augment must be differentiable"


def test_train_resume_render(dataset, tmp_path):
    """The whole path: train, checkpoint, resume, render from disk."""
    ck = tmp_path / "ck"
    common = {
        "size": 16,
        "channels": 32,
        "batch_size": 4,
        "ckpt_dir": ck,
        "log_every": 100,
        "save_every_steps": 2,
        "device": torch.device("cpu"),
    }
    gg.train(dataset, max_steps=2, **common)
    first = gg.latest_checkpoint(ck)
    assert first is not None and (ck / "generator.pt").exists()

    gg.train(dataset, max_steps=4, **common)
    blob = torch.load(gg.latest_checkpoint(ck), map_location="cpu", weights_only=False)
    assert blob["step"] == 4, blob["step"]

    out = gg.render_interpolation(
        ck / "generator.pt", tmp_path / "v.mp4", keys=3, frames=4, device=torch.device("cpu")
    )
    assert out.exists() and out.stat().st_size > 0


def test_empty_loader_is_rejected(tmp_path):
    """Fewer images than batch_size with drop_last leaves no batches at all.

    The loop used to spin on an empty loader forever, producing no output and
    no checkpoint.
    """
    d = tmp_path / "tiny" / "a"
    d.mkdir(parents=True)
    import numpy as np
    from PIL import Image

    for i in range(2):
        Image.fromarray(np.zeros((16, 16), dtype="uint8")).save(d / f"{i}.png")
    with pytest.raises(ValueError, match="no batches"):
        gg.train(
            tmp_path / "tiny",
            size=16,
            channels=32,
            batch_size=8,
            max_steps=2,
            ckpt_dir=tmp_path / "ck",
            device=torch.device("cpu"),
        )


def test_resume_with_epochs_makes_progress(dataset, tmp_path):
    """`epochs` counts passes from where this run starts, not from zero.

    Computed from zero, resuming with the same value left max_steps at or below
    the restored step and the run returned immediately.
    """
    ck = tmp_path / "ck"
    kw = {
        "size": 16,
        "channels": 32,
        "batch_size": 4,
        "ckpt_dir": ck,
        "device": torch.device("cpu"),
        "log_every": 99,
    }
    gg.train(dataset, epochs=1, **kw)
    first = torch.load(gg.latest_checkpoint(ck), map_location="cpu", weights_only=True)["step"]
    gg.train(dataset, epochs=1, **kw)
    second = torch.load(gg.latest_checkpoint(ck), map_location="cpu", weights_only=True)["step"]
    assert second > first, f"resume trained no steps: {first} -> {second}"
