"""Shape and wiring checks. Seconds to run, no dataset needed.

Runs under pytest, or directly as a script.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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


def test_train_step():
    dev = torch.device("cpu")
    gen, dis = gg.Generator(64).to(dev), gg.Discriminator(64).to(dev)
    og = torch.optim.Adam(gen.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(dis.parameters(), 2e-4, betas=(0.5, 0.999))
    x = torch.rand(2, 1, 64, 64) * 1.8 - 0.9
    ld, lg = gg.train_step(gen, dis, og, od, x, dev)
    assert ld == ld and lg == lg, "loss is NaN"


def test_slerp_norm():
    a, b = torch.randn(1, 200), torch.randn(1, 200)
    mid = gg.slerp(a, b, 0.5).norm().item()
    ends = (a.norm().item() + b.norm().item()) / 2
    # A linear midpoint collapses toward zero; a spherical one should not.
    assert mid > 0.9 * ends, f"slerp midpoint norm {mid:.2f} vs ends {ends:.2f}"


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


def _dataset(tmp_path, n=8, size=16):
    """A tiny ImageFolder-shaped dataset of random glyph-ish blobs."""
    import numpy as np
    from PIL import Image

    d = tmp_path / "data" / "a"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = np.zeros((size, size), dtype="uint8")
        img[4:12, 3 + (i % 3) : 9 + (i % 3)] = 255
        img = np.clip(img + rng.integers(0, 40, img.shape), 0, 255).astype("uint8")
        Image.fromarray(img).save(d / f"face{i}.png")
    return tmp_path / "data"


def test_loader_range(tmp_path):
    """Targets must land inside tanh's reachable range, not on its asymptotes.

    The generator ends in Tanh, which approaches +/-1 without arriving. A target
    of exactly 1.0 needs infinite pre-activation and returns no gradient, and
    scraped glyphs are almost entirely pure black and pure white.
    """
    loader = gg.make_loader(_dataset(tmp_path), size=16, batch_size=4, num_workers=0)
    x, _ = next(iter(loader))
    assert x.min() >= -gg.TARGET_SCALE - 1e-6, x.min()
    assert x.max() <= gg.TARGET_SCALE + 1e-6, x.max()
    # a fully black and fully white pixel must reach the ends, not fall short
    assert abs(x.min() + gg.TARGET_SCALE) < 1e-5
    assert abs(x.max() - gg.TARGET_SCALE) < 1e-5


def test_loader_yields_pairs(tmp_path):
    """ImageFolder yields (image, label); training must unpack, not iterate raw.

    Deleting the custom dataset class that silently dropped the label broke the
    training loop with `'list' object has no attribute 'to'`.
    """
    loader = gg.make_loader(_dataset(tmp_path), size=16, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert isinstance(batch, (list, tuple)) and len(batch) == 2, type(batch)
    assert isinstance(batch[0], torch.Tensor)


def test_train_resume_render(tmp_path):
    """The whole path: train, checkpoint, resume, render from disk."""
    data = _dataset(tmp_path)
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
    gg.train(data, max_steps=2, **common)
    first = gg.latest_checkpoint(ck)
    assert first is not None and (ck / "generator.pt").exists()

    gg.train(data, max_steps=4, **common)
    blob = torch.load(gg.latest_checkpoint(ck), map_location="cpu", weights_only=False)
    assert blob["step"] == 4, blob["step"]

    out = gg.render_interpolation(
        ck / "generator.pt", tmp_path / "v.mp4", keys=3, frames=4, device=torch.device("cpu")
    )
    assert out.exists() and out.stat().st_size > 0


def test_checkpoint_reads_pre_rename_keys(tmp_path):
    """Checkpoints written before the epoch->step and width->channels renames."""
    gen = gg.Generator(16, channels=32)
    path = tmp_path / "old.pt"
    torch.save(
        {"gen": gen.state_dict(), "size": 16, "width": 32, "latent": gg.LATENT, "epoch": 7}, path
    )
    loaded = gg.load_generator(path, torch.device("cpu"))
    assert loaded(torch.randn(1, gg.LATENT)).shape == (1, 1, 16, 16)


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


def test_to_image_maps_extremes():
    lo = gg.to_image(torch.full((1, 4, 4), -gg.TARGET_SCALE))
    hi = gg.to_image(torch.full((1, 4, 4), gg.TARGET_SCALE))
    assert lo.min() == lo.max() == 0, lo.min()
    assert hi.min() == hi.max() == 255, hi.max()


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


def test_stages_rejects_bad_size():
    for bad in (5, 6, 100, 2):
        try:
            gg._stages(bad)
        except ValueError:
            continue
        raise AssertionError(f"size {bad} should have been rejected")


def test_checkpoint_roundtrip(tmp_path):
    gen, dis = gg.Generator(64, channels=256), gg.Discriminator(64, channels=256)
    og = torch.optim.Adam(gen.parameters(), 2e-4)
    od = torch.optim.Adam(dis.parameters(), 2e-4)
    gg.save_checkpoint(tmp_path / "step0000000.pt", gen, dis, og, od, 0, 64, 256)
    assert (tmp_path / "generator.pt").exists(), "light generator checkpoint missing"
    loaded = gg.load_generator(tmp_path / "generator.pt", torch.device("cpu"))
    assert loaded(torch.randn(1, gg.LATENT)).shape == (1, 1, 64, 64)


if __name__ == "__main__":
    import tempfile

    tmp_path = Path(tempfile.mkdtemp())
    for fn in (
        test_shapes,
        test_channels,
        test_train_step,
        test_slerp_norm,
        test_labels_uniform,
        test_augment,
        test_accelerator,
        test_measures_on_known_shapes,
        test_to_image_maps_extremes,
        test_find_direction_recovers_a_planted_axis,
        test_stages_rejects_bad_size,
    ):
        fn()
        print(f"ok  {fn.__name__}")
    for fn in (
        test_loader_range,
        test_loader_yields_pairs,
        test_train_resume_render,
        test_checkpoint_reads_pre_rename_keys,
        test_checkpoint_roundtrip,
    ):
        fn(tmp_path)
        print(f"ok  {fn.__name__}")
    print("ok  test_checkpoint_roundtrip")
    print("\nall checks passed")
