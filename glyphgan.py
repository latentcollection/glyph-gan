"""DCGAN over rasterised glyphs, with latent-space interpolation.

Architecture follows Radford et al. (2015), scaled to the target resolution.
Implementation derived from the PyTorch DCGAN tutorial (BSD-3-Clause).

Datasets come from any folder of images laid out for `torchvision.ImageFolder`,
e.g. the output of `fontscrape --glyphs a --size 64 --mode xheight`.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

LATENT = 200
BASE = 4

# Channels at the 4x4 stage, halving outward. 1024 (the width the original
# DCGAN paper used against ~100k images) gives a 25M-parameter model that
# collapses on a few hundred glyphs - it memorises rather than generalises, and
# the generator stops varying with z at all. Raise it as the dataset grows.
DEFAULT_WIDTH = 256

# Targets land at +/-0.9 rather than +/-1. tanh only approaches its asymptotes,
# so a target of exactly 1.0 demands infinite pre-activation and returns zero
# gradient; at 0.9 the generator still sees 19% of the maximum. Scraped glyphs
# are almost entirely pure black and pure white, so nearly every target pixel
# would otherwise sit on an unreachable value.
TARGET_SCALE = 0.9


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stages(size: int, width: int = DEFAULT_WIDTH) -> list[int]:
    """Channel width at each 2x stage, halving down from `width`."""
    if size < BASE * 2 or size & (size - 1):
        raise ValueError(f"size must be a power of two >= {BASE * 2}, got {size}")
    n = int(math.log2(size // BASE))
    return [max(width >> i, 32) for i in range(n)]


class Generator(nn.Module):
    def __init__(self, size: int = 64, latent: int = LATENT, width: int = DEFAULT_WIDTH,
                 alpha: float = 0.2):
        super().__init__()
        ch = _stages(size, width)
        self.input = nn.Linear(latent, BASE * BASE * ch[0])

        layers: list[nn.Module] = [nn.BatchNorm2d(ch[0]), nn.LeakyReLU(alpha)]
        for i, c in enumerate(ch):
            out = ch[i + 1] if i + 1 < len(ch) else 1
            layers.append(nn.ConvTranspose2d(c, out, 4, 2, 1))
            if out == 1:
                layers.append(nn.Tanh())
            else:
                layers += [nn.BatchNorm2d(out), nn.LeakyReLU(alpha)]
        self.net = nn.Sequential(*layers)
        self.ch0 = ch[0]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.input(z).view(-1, self.ch0, BASE, BASE))


class Discriminator(nn.Module):
    def __init__(self, size: int = 64, width: int = DEFAULT_WIDTH, alpha: float = 0.2):
        super().__init__()
        ch = list(reversed(_stages(size, width)))

        layers: list[nn.Module] = []
        prev = 1
        for i, c in enumerate(ch):
            layers.append(nn.Conv2d(prev, c, 4, 2, 1))
            # No norm on the first layer; it would wash out the input statistics.
            if i:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(alpha))
            prev = c
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(BASE * BASE * prev, 1)
        self.flat = BASE * BASE * prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(x).reshape(-1, self.flat))


def make_loader(img_dir, size=64, batch_size=64, num_workers=2) -> DataLoader:
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((size, size), transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5 / TARGET_SCALE,)),
    ])
    data = datasets.ImageFolder(str(img_dir), tf)
    return DataLoader(data, batch_size=batch_size, shuffle=True,
                      drop_last=True, num_workers=num_workers)


def _soft_labels(shape, device, flip_rate=0.03, smooth=0.1):
    """Smoothed targets with a few labels swapped, per ganhacks.

    Both tricks slow the discriminator down so it cannot win outright early and
    starve the generator of gradient.
    """
    real = torch.empty(shape, device=device).uniform_(1 - smooth, 1 + smooth)
    fake = torch.empty(shape, device=device).uniform_(0.0, 2 * smooth)
    flip = torch.rand(shape, device=device) < flip_rate
    return torch.where(flip, fake, real), torch.where(flip, real, fake)


def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _contrast(x):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + m


def _translate(x, ratio=0.125):
    n, _, h, w = x.shape
    pad = max(1, int(h * ratio + 0.5))
    p = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (n,))
    ox = torch.randint(0, 2 * pad + 1, (n,))
    return torch.stack([p[i, :, oy[i]:oy[i] + h, ox[i]:ox[i] + w] for i in range(n)])


def _cutout(x, ratio=0.5):
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5) // 2, int(w * ratio + 0.5) // 2
    cy = torch.randint(0, h, (n,))
    cx = torch.randint(0, w, (n,))
    mask = torch.ones_like(x)
    for i in range(n):
        mask[i, :, max(0, cy[i] - ch):cy[i] + ch, max(0, cx[i] - cw):cx[i] + cw] = 0
    return x * mask


def diff_augment(x):
    """Differentiable augmentation applied to real and fake alike.

    With only a few thousand glyphs the discriminator memorises the set and
    stops handing the generator a useful gradient. Augmenting both sides with
    operations gradients can flow through widens the effective dataset without
    teaching the generator to reproduce the augmentations themselves.

    Saturation, part of the usual policy, is omitted: these are single-channel
    images, so it is a no-op.
    """
    return _cutout(_translate(_contrast(_brightness(x))))


def train_step(gen, dis, opt_g, opt_d, real, device, augment=False):
    real = real.to(device)
    batch = real.size(0)
    aug = diff_augment if augment else (lambda t: t)

    opt_d.zero_grad(set_to_none=True)
    pred_real = dis(aug(real))
    y_real, y_fake = _soft_labels(pred_real.shape, device)
    fake = gen(torch.randn(batch, LATENT, device=device))
    loss_d = (F.binary_cross_entropy_with_logits(pred_real, y_real)
              + F.binary_cross_entropy_with_logits(dis(aug(fake.detach())), y_fake))
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    pred_fake = dis(aug(fake))
    loss_g = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation between two latents.

    A straight line between two Gaussian samples dips through norms the model
    never saw in training, so linearly interpolated morphs sag and wash out
    halfway. Travelling along the hypersphere keeps ||z|| roughly constant.
    """
    a_n = a / a.norm(dim=-1, keepdim=True)
    b_n = b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((a_n * b_n).sum(-1, keepdim=True).clamp(-1.0, 1.0))
    sin_omega = torch.sin(omega)
    if bool((sin_omega.abs() < 1e-6).all()):
        return torch.lerp(a, b, t)
    return (torch.sin((1.0 - t) * omega) / sin_omega) * a + (torch.sin(t * omega) / sin_omega) * b


def to_image(x: torch.Tensor):
    """Single generator output to an 8-bit grayscale frame."""
    x = x.detach().float().clamp(-TARGET_SCALE, TARGET_SCALE) / TARGET_SCALE
    return ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze().cpu().numpy()


def load_generator(checkpoint, device=None):
    device = device or pick_device()
    blob = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    gen = Generator(size=blob.get("size", 64), latent=blob.get("latent", LATENT),
                    width=blob.get("width", DEFAULT_WIDTH))
    gen.load_state_dict(blob["gen"])
    return gen.to(device).eval()


def render_interpolation(generator, out, keys=8, frames=60, fps=30,
                         loop=True, seed=None, device=None):
    """Walk the latent space through `keys` waypoints and write a video.

    `generator` may be a live module or a path to a checkpoint. Nothing here
    depends on training state, so a video can be rendered from any checkpoint
    at any time.
    """
    device = device or pick_device()
    gen = generator if isinstance(generator, nn.Module) else load_generator(generator, device)
    gen = gen.to(device).eval()

    if isinstance(keys, int):
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        latent = gen.input.in_features
        keys = [torch.randn(1, latent, generator=g).to(device) for _ in range(keys)]
    if loop:
        keys = list(keys) + [keys[0]]

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("imageio is required to render video") from exc

    video = []
    with torch.no_grad():
        for a, b in zip(keys, keys[1:]):
            for k in range(frames):
                video.append(to_image(gen(slerp(a, b, k / frames))[0]))

    out = Path(out)
    try:
        imageio.mimwrite(out, video, fps=fps)
    except Exception as exc:
        raise RuntimeError(
            f"could not write {out}. mp4 output needs the ffmpeg plugin: "
            "pip install imageio-ffmpeg"
        ) from exc
    return out


def save_checkpoint(path, gen, dis, opt_g, opt_d, epoch, size, width=DEFAULT_WIDTH, keep_last=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "gen": gen.state_dict(),
        "dis": dis.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "size": size,
        "width": width,
        "latent": LATENT,
    }, path)

    # Rendering needs the generator alone, so keep a light copy next to the
    # full checkpoints - roughly a fifth the size, and the only file worth
    # keeping once a run is finished.
    save_generator(path.parent / "generator.pt", gen, size, width)

    # Full checkpoints carry both models and both Adam states, so they run to
    # a few hundred MB each. Keep only the most recent few.
    if keep_last:
        stale = sorted(path.parent.glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
        for old in stale[:-keep_last]:
            old.unlink()
    return path


def save_generator(path, gen, size, width=DEFAULT_WIDTH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gen": gen.state_dict(), "size": size, "width": width,
                "latent": LATENT}, path)
    return path


def latest_checkpoint(ckpt_dir):
    found = sorted(Path(ckpt_dir).glob("epoch*.pt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def train(img_dir, size=64, epochs=50, batch_size=32, lr=2e-4, betas=(0.5, 0.999),
          width=DEFAULT_WIDTH, augment=True, ckpt_dir="checkpoints", save_every=5,
          resume=True, device=None, num_workers=None, log_every=20):
    """Train to `epochs`, checkpointing as it goes.

    Every exit path writes a checkpoint, including Ctrl-C and a crash, because
    an unsaved generator is a run you cannot render a video from.
    """
    device = device or pick_device()
    if num_workers is None:
        # Worker processes spawn rather than fork off CUDA, and on macOS that
        # deadlocks inside notebooks often enough not to be worth the speedup.
        num_workers = 2 if device.type == "cuda" else 0

    loader = make_loader(img_dir, size=size, batch_size=batch_size, num_workers=num_workers)
    gen, dis = Generator(size, width=width).to(device), Discriminator(size, width=width).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr, betas=betas)
    opt_d = torch.optim.Adam(dis.parameters(), lr, betas=betas)

    start = 0
    if resume:
        found = latest_checkpoint(ckpt_dir)
        if found:
            blob = torch.load(found, map_location=device, weights_only=False)
            gen.load_state_dict(blob["gen"])
            dis.load_state_dict(blob["dis"])
            opt_g.load_state_dict(blob["opt_g"])
            opt_d.load_state_dict(blob["opt_d"])
            start = blob["epoch"] + 1
            print(f"resumed {found.name} at epoch {start}")

    n = len(loader)
    params = sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in dis.parameters())
    print(f"{device.type} | {size}px | width {width} | {params/1e6:.1f}M params | "
          f"batch {batch_size} | {len(loader.dataset)} images | {n} steps/epoch | "
          f"epochs {start}-{epochs - 1}")

    epoch = start
    try:
        for epoch in range(start, epochs):
            for i, (x, _) in enumerate(loader, 1):
                loss_d, loss_g = train_step(gen, dis, opt_g, opt_d, x, device, augment)
                if i % log_every == 0 or i == n:
                    print(f"epoch {epoch:3d}  {i:4d}/{n}  loss_d {loss_d:.3f}  loss_g {loss_g:.3f}")
            if save_every and (epoch + 1) % save_every == 0:
                save_checkpoint(Path(ckpt_dir) / f"epoch{epoch:04d}.pt",
                                gen, dis, opt_g, opt_d, epoch, size, width)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        path = save_checkpoint(Path(ckpt_dir) / f"epoch{epoch:04d}.pt",
                               gen, dis, opt_g, opt_d, epoch, size, width)
        print(f"saved {path}")

    return gen, dis
