"""Checkpoints round-trip, and older ones still load."""

import torch

import glyphgan as gg


def test_checkpoint_roundtrip(tmp_path):
    gen, dis = gg.Generator(64, channels=256), gg.Discriminator(64, channels=256)
    og = torch.optim.Adam(gen.parameters(), 2e-4)
    od = torch.optim.Adam(dis.parameters(), 2e-4)
    gg.save_checkpoint(tmp_path / "step0000000.pt", gen, dis, og, od, 0, 64, 256)
    assert (tmp_path / "generator.pt").exists(), "light generator checkpoint missing"
    loaded = gg.load_generator(tmp_path / "generator.pt", torch.device("cpu"))
    assert loaded(torch.randn(1, gg.LATENT)).shape == (1, 1, 64, 64)


def test_checkpoint_reads_pre_rename_keys(tmp_path):
    """Checkpoints written before the epoch->step and width->channels renames."""
    gen = gg.Generator(16, channels=32)
    path = tmp_path / "old.pt"
    torch.save(
        {"gen": gen.state_dict(), "size": 16, "width": 32, "latent": gg.LATENT, "epoch": 7}, path
    )
    loaded = gg.load_generator(path, torch.device("cpu"))
    assert loaded(torch.randn(1, gg.LATENT)).shape == (1, 1, 16, 16)
