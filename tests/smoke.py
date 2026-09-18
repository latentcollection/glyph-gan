"""Shape and wiring checks. Runs on CPU in seconds; no dataset needed."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import glyphgan as gg


def check_shapes():
    for size in (64, 128):
        gen, dis = gg.Generator(size), gg.Discriminator(size)
        out = gen(torch.randn(2, gg.LATENT))
        assert out.shape == (2, 1, size, size), f"{size}: {out.shape}"
        assert dis(out).shape == (2, 1), f"{size}: {dis(out).shape}"
        assert out.min() >= -1.0 and out.max() <= 1.0, "tanh range"


def check_width():
    small = gg.Generator(64, width=256)
    big = gg.Generator(64, width=1024)
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in big.parameters())


def check_train_step():
    dev = torch.device("cpu")
    gen, dis = gg.Generator(64).to(dev), gg.Discriminator(64).to(dev)
    og = torch.optim.Adam(gen.parameters(), 2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(dis.parameters(), 2e-4, betas=(0.5, 0.999))
    x = torch.rand(2, 1, 64, 64) * 1.8 - 0.9
    ld, lg = gg.train_step(gen, dis, og, od, x, dev)
    assert ld == ld and lg == lg, "loss is NaN"


def check_slerp_norm():
    a, b = torch.randn(1, 200), torch.randn(1, 200)
    mid = gg.slerp(a, b, 0.5).norm().item()
    ends = (a.norm().item() + b.norm().item()) / 2
    # A linear midpoint collapses toward zero; a spherical one should not.
    assert mid > 0.9 * ends, f"slerp midpoint norm {mid:.2f} vs ends {ends:.2f}"


def check_labels_uniform():
    counts = torch.zeros(8)
    for _ in range(2000):
        real, _ = gg._soft_labels((8, 1), torch.device("cpu"), flip_rate=0.03)
        counts += (real.squeeze() < 0.5).float()
    rate = counts / 2000
    assert rate.max() < 0.08, f"flips not uniform across the batch: {rate.tolist()}"


def check_augment():
    x = (torch.rand(4, 1, 64, 64) * 1.8 - 0.9).requires_grad_(True)
    y = gg.diff_augment(x)
    assert y.shape == x.shape, y.shape
    y.sum().backward()
    assert x.grad.abs().sum() > 0, "augment must be differentiable"


def check_checkpoint_roundtrip(tmp):
    gen, dis = gg.Generator(64, width=256), gg.Discriminator(64, width=256)
    og = torch.optim.Adam(gen.parameters(), 2e-4)
    od = torch.optim.Adam(dis.parameters(), 2e-4)
    gg.save_checkpoint(tmp / "epoch0000.pt", gen, dis, og, od, 0, 64, 256)
    assert (tmp / "generator.pt").exists(), "light generator checkpoint missing"
    loaded = gg.load_generator(tmp / "generator.pt", torch.device("cpu"))
    assert loaded(torch.randn(1, gg.LATENT)).shape == (1, 1, 64, 64)


if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    for fn in (check_shapes, check_width, check_train_step, check_slerp_norm,
               check_labels_uniform, check_augment):
        fn()
        print(f"ok  {fn.__name__}")
    check_checkpoint_roundtrip(tmp)
    print("ok  check_checkpoint_roundtrip")
    print("\nall checks passed")
