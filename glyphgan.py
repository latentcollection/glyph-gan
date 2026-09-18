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
MAX_CH = 1024

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


def _stages(size: int) -> list[int]:
    """Channel width at each 2x stage, halving down from MAX_CH."""
    if size < BASE * 2 or size & (size - 1):
        raise ValueError(f"size must be a power of two >= {BASE * 2}, got {size}")
    n = int(math.log2(size // BASE))
    return [max(MAX_CH >> i, 32) for i in range(n)]


class Generator(nn.Module):
    def __init__(self, size: int = 64, latent: int = LATENT, alpha: float = 0.2):
        super().__init__()
        ch = _stages(size)
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
    def __init__(self, size: int = 64, alpha: float = 0.2):
        super().__init__()
        ch = list(reversed(_stages(size)))

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


def train_step(gen, dis, opt_g, opt_d, real, device):
    real = real.to(device)
    batch = real.size(0)

    opt_d.zero_grad(set_to_none=True)
    pred_real = dis(real)
    y_real, y_fake = _soft_labels(pred_real.shape, device)
    fake = gen(torch.randn(batch, LATENT, device=device))
    loss_d = (F.binary_cross_entropy_with_logits(pred_real, y_real)
              + F.binary_cross_entropy_with_logits(dis(fake.detach()), y_fake))
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    pred_fake = dis(fake)
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
    gen = Generator(size=blob.get("size", 64), latent=blob.get("latent", LATENT))
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
