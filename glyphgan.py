"""DCGAN over rasterised glyphs, with latent-space interpolation.

Architecture follows Radford et al. (2015), scaled to the target resolution.
Implementation derived from the PyTorch DCGAN tutorial (BSD-3-Clause).

Datasets come from any folder of images laid out for `torchvision.ImageFolder`,
e.g. the output of `fontscrape --glyphs a --size 64 --mode xheight`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sized
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- types and constants -------------------------------------------------------


class HasStateDict(Protocol):
    """Anything whose weights can be written to a checkpoint.

    Generator and EMA both qualify, which is what lets a run checkpoint the
    averaged weights without the caller knowing which one it is holding.
    """

    def state_dict(self) -> dict: ...


# A live generator, or a path to a checkpoint holding one.
GeneratorSource = nn.Module | str | Path

Measure = Callable[[torch.Tensor], torch.Tensor]

LATENT = 200

BASE = 4

# Channels at the 4x4 stage, halving outward. 1024 (what the original
# DCGAN paper used against ~100k images) gives a 25M-parameter model that
# collapses on a few hundred glyphs - it memorises rather than generalises, and
# the generator stops varying with z at all. Raise it as the dataset grows.
DEFAULT_CHANNELS = 256

# Targets land at +/-0.9 rather than +/-1. tanh only approaches its asymptotes,
# so a target of exactly 1.0 demands infinite pre-activation and returns zero
# gradient; at 0.9 the generator still sees 19% of the maximum. Scraped glyphs
# are almost entirely pure black and pure white, so nearly every target pixel
# would otherwise sit on an unreachable value.
TARGET_SCALE = 0.9


# --- helpers -------------------------------------------------------------------


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(generator: GeneratorSource, device: torch.device) -> nn.Module:
    """A live module or a checkpoint path, either way on `device`.

    Deliberately does not change training mode - the caller may have handed us
    a generator that is mid-run, and leaving it in eval() would silently freeze
    its BatchNorm for the rest of training. Use `_sampling` for that.
    """
    gen = generator if isinstance(generator, nn.Module) else load_generator(generator, device)
    return gen.to(device)


@contextmanager
def _sampling(gen: nn.Module):
    """Evaluate `gen`, then restore whatever mode the caller had it in."""
    was_training = gen.training
    gen.eval()
    try:
        yield gen
    finally:
        gen.train(was_training)


def _latent_of(gen: nn.Module) -> int:
    """A generator's own latent size, rather than assuming the module default."""
    inp = getattr(gen, "input", None)
    return getattr(inp, "in_features", LATENT)


def _write_video(frames: list[np.ndarray], out: str | Path, fps: int) -> Path:
    if not frames:
        raise ValueError("no frames to write")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "writing video needs imageio: pip install imageio imageio-ffmpeg"
        ) from exc

    out = _ensure_parent(out)
    try:
        imageio.mimwrite(out, np.stack(frames), fps=fps)
    except Exception as exc:
        raise RuntimeError(
            f"could not write {out}. mp4 output needs the ffmpeg plugin: pip install imageio-ffmpeg"
        ) from exc
    return out


# --- models --------------------------------------------------------------------


def _init_weights(module: nn.Module) -> None:
    """DCGAN draws conv and batchnorm weights from N(0, 0.02).

    PyTorch defaults to Kaiming-uniform, a different distribution entirely. The
    paper's choice is load-bearing rather than incidental: DCGAN is unstable at
    other initialisations, and the reference implementation applies this.
    """
    name = type(module).__name__
    if "Conv" in name or "Linear" in name:
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def _stages(size: int, channels: int = DEFAULT_CHANNELS) -> list[int]:
    """Feature count at each 2x stage, halving down from `channels`."""
    if size < BASE * 2 or size & (size - 1):
        raise ValueError(f"size must be a power of two >= {BASE * 2}, got {size}")
    n = int(math.log2(size // BASE))
    return [max(channels >> i, 32) for i in range(n)]


class Generator(nn.Module):
    def __init__(
        self,
        size: int = 64,
        latent: int = LATENT,
        channels: int = DEFAULT_CHANNELS,
    ):
        super().__init__()
        ch = _stages(size, channels)
        self.input = nn.Linear(latent, BASE * BASE * ch[0])

        # ReLU throughout the generator and Tanh at the output, per the paper.
        # LeakyReLU is the discriminator's activation, not the generator's.
        layers: list[nn.Module] = [nn.BatchNorm2d(ch[0]), nn.ReLU(True)]
        for i, c in enumerate(ch):
            out = ch[i + 1] if i + 1 < len(ch) else 1
            layers.append(nn.ConvTranspose2d(c, out, 4, 2, 1))
            if out == 1:
                layers.append(nn.Tanh())
            else:
                layers += [nn.BatchNorm2d(out), nn.ReLU(True)]
        self.net = nn.Sequential(*layers)
        self.ch0 = ch[0]
        self.apply(_init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.input(z).view(-1, self.ch0, BASE, BASE))


class Discriminator(nn.Module):
    def __init__(
        self,
        size: int = 64,
        channels: int = DEFAULT_CHANNELS,
        alpha: float = 0.2,
        spectral: bool = True,
    ):
        super().__init__()
        ch = list(reversed(_stages(size, channels)))

        # Spectral normalisation constrains each layer's Lipschitz constant and
        # replaces batchnorm rather than joining it. Batchnorm couples samples
        # within a batch, which at batch 32 makes the discriminator's statistics
        # noisy and lets information leak between real and fake; this is the one
        # place the model departs from the 2015 DCGAN paper, and it costs about
        # 19% more per step.
        norm = spectral_norm if spectral else (lambda m: m)

        layers: list[nn.Module] = []
        prev = 1
        for i, c in enumerate(ch):
            layers.append(norm(nn.Conv2d(prev, c, 4, 2, 1)))
            # No norm on the first layer; it would wash out the input statistics.
            if i and not spectral:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.LeakyReLU(alpha))
            prev = c
        self.net = nn.Sequential(*layers)
        self.output = norm(nn.Linear(BASE * BASE * prev, 1))
        self.flat = BASE * BASE * prev
        if not spectral:
            self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(x).reshape(-1, self.flat))


class EMA:
    """Exponential moving average of the generator's weights.

    The raw generator chases the discriminator from step to step, so its
    samples flicker. An averaged copy sits in the middle of that oscillation
    and renders markedly cleaner - which matters more here than usual, because
    the output is a video where frame-to-frame instability is visible directly.

    `decay` ramps in over the first steps; at a fixed 0.999 the average would
    still be mostly its initialisation thousands of steps in.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.step = 0
        sd = model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in sd.items()}
        self._float = [k for k, v in sd.items() if v.dtype.is_floating_point]
        self._other = [k for k in sd if k not in set(self._float)]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        cur = model.state_dict()
        # A fused torch._foreach_mul_/_add_ would collapse these launches, but
        # MPS implements neither, so the per-tensor loop is the portable form.
        for k in self._float:
            self.shadow[k].mul_(d).add_(cur[k], alpha=1 - d)
        for k in self._other:
            self.shadow[k] = cur[k].detach().clone()

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}

    def to_generator(self, size: int, channels: int, device: torch.device) -> Generator:
        """A Generator carrying the averaged weights, ready to sample from."""
        gen = Generator(size, channels=channels).to(device)
        gen.load_state_dict(self.state_dict())
        return gen.eval()


# --- data ----------------------------------------------------------------------


class _Cached(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    """Decode the dataset once and keep it in memory.

    A glyph set is small - a few thousand 64px greyscale images is tens of MB -
    but it is otherwise re-read and re-decoded from PNG on every step, which
    costs a real slice of a step already dominated by per-op overhead rather
    than by arithmetic.
    """

    def __init__(self, base: datasets.ImageFolder) -> None:
        self.tensors = torch.stack([base[i][0] for i in range(len(base))])

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.tensors[index], 0


def make_loader(
    img_dir: str | Path,
    size: int = 64,
    batch_size: int = 64,
    num_workers: int = 2,
    cache: bool = True,
) -> DataLoader:
    tf = transforms.Compose(
        [
            transforms.Grayscale(1),
            transforms.Resize((size, size), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5 / TARGET_SCALE,)),
        ]
    )
    data = datasets.ImageFolder(str(img_dir), tf)
    if cache:
        data = _Cached(data)
        num_workers = 0  # nothing left to overlap; workers would only add IPC
    return DataLoader(
        data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
    )


# --- augmentation --------------------------------------------------------------


def _brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _contrast(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + m


def _translate(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Random per-sample shift, vectorised.

    Uses gather-style indexing rather than grid_sample: MPS implements neither
    border padding nor grid_sampler's backward pass, and the augment has to be
    differentiable for DiffAugment to mean anything.
    """
    n, _, h, w = x.shape
    pad = max(1, int(h * ratio + 0.5))
    bg = -TARGET_SCALE
    p = F.pad(x - bg, (pad, pad, pad, pad)) + bg

    oy = torch.randint(0, 2 * pad + 1, (n, 1, 1), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (n, 1, 1), device=x.device)
    ys = oy + torch.arange(h, device=x.device).view(1, h, 1)
    xs = ox + torch.arange(w, device=x.device).view(1, 1, w)
    b = torch.arange(n, device=x.device).view(n, 1, 1)
    return p[b, :, ys, xs].permute(0, 3, 1, 2).contiguous()


def _cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5) // 2, int(w * ratio + 0.5) // 2
    cy = torch.randint(0, h, (n, 1, 1), device=x.device)
    cx = torch.randint(0, w, (n, 1, 1), device=x.device)
    ys = torch.arange(h, device=x.device).view(1, h, 1)
    xs = torch.arange(w, device=x.device).view(1, 1, w)
    keep = ((ys - cy).abs() >= ch) | ((xs - cx).abs() >= cw)
    return x * keep.unsqueeze(1).to(x.dtype)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """Differentiable augmentation applied to real and fake alike.

    With only a few thousand glyphs the discriminator memorises the set and
    stops handing the generator a useful gradient. Augmenting both sides with
    operations gradients can flow through widens the effective dataset without
    teaching the generator to reproduce the augmentations themselves.

    Saturation, part of the usual policy, is omitted: these are single-channel
    images, so it is a no-op.
    """
    return _cutout(_translate(_contrast(_brightness(x))))


# --- checkpoints ---------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    gen: nn.Module,
    dis: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    step: int,
    size: int,
    channels: int = DEFAULT_CHANNELS,
    keep_last: int = 3,
    ema: EMA | None = None,
) -> Path:
    path = _ensure_parent(path)
    torch.save(
        {
            "gen": gen.state_dict(),
            "dis": dis.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "step": step,
            "size": size,
            "channels": channels,
            "latent": LATENT,
            "ema": ema.state_dict() if ema is not None else None,
        },
        path,
    )

    # Rendering needs the generator alone, so keep a light copy next to the
    # full checkpoints - roughly a fifth the size, and the only file worth
    # keeping once a run is finished.
    save_generator(path.parent / "generator.pt", ema if ema is not None else gen, size, channels)

    # Full checkpoints carry both models and both Adam states, so they run to
    # a few hundred MB each. Keep only the most recent few.
    if keep_last:
        stale = sorted(path.parent.glob("step*.pt"), key=lambda p: p.stat().st_mtime)
        for old in stale[:-keep_last]:
            old.unlink()
    return path


def save_generator(
    path: str | Path, gen: HasStateDict, size: int, channels: int = DEFAULT_CHANNELS
) -> Path:
    """`gen` may be a Generator or an EMA wrapper - both expose state_dict()."""
    path = _ensure_parent(path)
    torch.save(
        {"gen": gen.state_dict(), "size": size, "channels": channels, "latent": LATENT}, path
    )
    return path


def load_generator(checkpoint: str | Path, device: torch.device | None = None) -> nn.Module:
    device = device or pick_device()
    blob = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    gen = Generator(
        size=blob.get("size", 64),
        latent=blob.get("latent", LATENT),
        # Checkpoints written before the rename store "width".
        channels=blob.get("channels", blob.get("width", DEFAULT_CHANNELS)),
    )
    gen.load_state_dict(blob["gen"])
    return gen.to(device).eval()


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    found = sorted(Path(ckpt_dir).glob("step*.pt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def save_preview(
    path: str | Path,
    gen: nn.Module,
    n: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
) -> Path:
    """Contact sheet of `n` samples, for watching a run progress."""
    from PIL import Image

    device = device or pick_device()
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, _latent_of(gen), generator=g).to(device)
    with _sampling(gen), torch.no_grad():
        sheet = np.hstack([to_image(v) for v in gen(z)])
    path = _ensure_parent(path)
    Image.fromarray(sheet).save(path)
    return path


# --- training ------------------------------------------------------------------


def _soft_labels(
    shape: tuple, device: torch.device, flip_rate: float = 0.03, smooth: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoothed targets with a few labels swapped, per ganhacks.

    Both tricks slow the discriminator down so it cannot win outright early and
    starve the generator of gradient.
    """
    real = torch.empty(shape, device=device).uniform_(1 - smooth, 1 + smooth)
    fake = torch.empty(shape, device=device).uniform_(0.0, 2 * smooth)
    flip = torch.rand(shape, device=device) < flip_rate
    return torch.where(flip, fake, real), torch.where(flip, real, fake)


def train_step(
    gen: nn.Module,
    dis: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    real: torch.Tensor,
    device: torch.device,
    augment: bool = False,
) -> tuple[float, float]:
    real = real.to(device)
    batch = real.size(0)
    aug = diff_augment if augment else (lambda t: t)

    opt_d.zero_grad(set_to_none=True)
    pred_real = dis(aug(real))
    y_real, y_fake = _soft_labels(pred_real.shape, device)
    fake = gen(torch.randn(batch, LATENT, device=device))
    loss_d = F.binary_cross_entropy_with_logits(
        pred_real, y_real
    ) + F.binary_cross_entropy_with_logits(dis(aug(fake.detach())), y_fake)
    loss_d.backward()
    opt_d.step()

    opt_g.zero_grad(set_to_none=True)
    pred_fake = dis(aug(fake))
    loss_g = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
    loss_g.backward()
    opt_g.step()

    return loss_d.item(), loss_g.item()


def train(
    img_dir: str | Path,
    size: int = 64,
    max_steps: int = 20000,
    epochs: int | None = None,
    batch_size: int = 32,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.5, 0.999),
    channels: int = DEFAULT_CHANNELS,
    augment: bool = True,
    spectral: bool = True,
    ckpt_dir: str | Path = "checkpoints",
    save_every_steps: int = 2000,
    preview_every: int | None = None,
    resume: bool = True,
    device: torch.device | None = None,
    num_workers: int | None = None,
    log_every: int = 100,
) -> tuple[nn.Module, nn.Module]:
    """Train for `max_steps` optimiser steps, checkpointing as it goes.

    The budget is in steps rather than epochs on purpose. A few hundred glyphs
    make an epoch only a handful of steps, so an epoch count borrowed from a
    large dataset trains for almost no time - 50 epochs of 272 images is 400
    steps, against the tens of thousands a GAN needs. Pass `epochs` instead only
    when you specifically want a pass count.

    Every exit path writes a checkpoint, including Ctrl-C and a crash, because
    an unsaved generator is a run you cannot render a video from.
    """
    device = device or pick_device()
    # make_loader caches the dataset in memory and then has nothing for worker
    # processes to overlap, so this stays 0 unless caching is turned off.
    if num_workers is None:
        num_workers = 0

    loader = make_loader(img_dir, size=size, batch_size=batch_size, num_workers=num_workers)
    n = len(loader)
    if n == 0:
        raise ValueError(
            f"{len(cast(Sized, loader.dataset))} images at batch_size {batch_size} with "
            "drop_last leaves no batches; lower batch_size or add images"
        )
    gen = Generator(size, channels=channels).to(device)
    dis = Discriminator(size, channels=channels, spectral=spectral).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr, betas=betas)
    opt_d = torch.optim.Adam(dis.parameters(), lr, betas=betas)

    start = 0
    if resume:
        found = latest_checkpoint(ckpt_dir)
        if found:
            blob = torch.load(found, map_location=device, weights_only=True)
            gen.load_state_dict(blob["gen"])
            dis.load_state_dict(blob["dis"])
            # Checkpoints written before the key was renamed store "epoch",
            # though the value was always a step count.
            start = blob.get("step", blob.get("epoch", 0))
            opt_g.load_state_dict(blob["opt_g"])
            opt_d.load_state_dict(blob["opt_d"])
            print(f"resumed {found.name} at step {start}")

    # Built after any restore, so the shadow starts from the restored weights
    # rather than from random initialisation.
    ema = EMA(gen)
    if resume:
        found = latest_checkpoint(ckpt_dir)
        if found:
            blob = torch.load(found, map_location=device, weights_only=True)
            if blob.get("ema"):
                ema.shadow = {k: v.to(device) for k, v in blob["ema"].items()}
                ema.step = start

    if epochs is not None:
        # A pass count is relative to where this run starts, so resuming with
        # the same `epochs` trains that many more passes rather than none.
        max_steps = start + epochs * n
    params = sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in dis.parameters())
    print(
        f"{device.type} | {size}px | ch {channels} | {params / 1e6:.1f}M params | "
        f"batch {batch_size} | {len(cast(Sized, loader.dataset))} images | {n} steps/epoch | "
        f"{max_steps} steps"
    )

    ckpt_dir = Path(ckpt_dir)

    def checkpoint(step: int) -> Path:
        return save_checkpoint(
            ckpt_dir / f"step{step:07d}.pt", gen, dis, opt_g, opt_d, step, size, channels, ema=ema
        )

    step = start
    try:
        while step < max_steps:
            for x, _ in loader:
                loss_d, loss_g = train_step(gen, dis, opt_g, opt_d, x, device, augment)
                ema.update(gen)
                step += 1
                if step % log_every == 0:
                    print(f"step {step:6d}/{max_steps}  loss_d {loss_d:.3f}  loss_g {loss_g:.3f}")
                if preview_every and step % preview_every == 0:
                    # Preview the averaged weights, since that is what renders.
                    save_preview(
                        ckpt_dir / f"preview{step:07d}.png",
                        ema.to_generator(size, channels, device),
                        device=device,
                    )
                if save_every_steps and step % save_every_steps == 0:
                    checkpoint(step)
                if step >= max_steps:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print(f"saved {checkpoint(step)} at step {step}")

    return gen, dis


# --- rendering -----------------------------------------------------------------


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


def to_image(x: torch.Tensor) -> np.ndarray:
    """Single generator output to an 8-bit greyscale frame."""
    x = x.detach().float().clamp(-TARGET_SCALE, TARGET_SCALE) / TARGET_SCALE
    return ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).squeeze().cpu().numpy()


def render_interpolation(
    generator: GeneratorSource,
    out: str | Path,
    keys: int | list[torch.Tensor] = 8,
    frames: int = 60,
    fps: int = 30,
    loop: bool = True,
    seed: int | None = None,
    device: torch.device | None = None,
) -> Path:
    """Walk the latent space through `keys` waypoints and write a video.

    `generator` may be a live module or a path to a checkpoint. Nothing here
    depends on training state, so a video can be rendered from any checkpoint
    at any time.
    """
    if frames < 1:
        raise ValueError(f"frames must be at least 1, got {frames}")
    device = device or pick_device()
    gen = _resolve(generator, device)

    if isinstance(keys, int):
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        keys = [torch.randn(1, _latent_of(gen), generator=g).to(device) for _ in range(keys)]
    keys = list(keys)
    if len(keys) < 2:
        raise ValueError("interpolation needs at least two waypoints")
    if loop:
        keys = keys + [keys[0]]

    video = []
    with _sampling(gen), torch.no_grad():
        for a, b in zip(keys, keys[1:], strict=False):
            video += [to_image(gen(slerp(a, b, k / frames))[0]) for k in range(frames)]
    return _write_video(video, out, fps)


def render_direction(
    gen: GeneratorSource,
    direction: torch.Tensor,
    out: str | Path,
    span: float = 3.0,
    frames: int = 60,
    fps: int = 30,
    seed: int = 0,
    device: torch.device | None = None,
    base: torch.Tensor | None = None,
) -> Path:
    """Walk a single latent direction from -span to +span and write a video.

    Everything except the chosen direction is held fixed, so the result reads as one
    typeface being pushed along that axis rather than as a dissolve between two
    unrelated letterforms.
    """
    if frames < 2:
        raise ValueError(f"walking an axis needs at least 2 frames, got {frames}")
    device = device or pick_device()
    gen = _resolve(gen, device)
    if base is None:
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(1, _latent_of(gen), generator=g)
    base = base.to(device)
    d = direction.to(device).unsqueeze(0)

    video = []
    with _sampling(gen), torch.no_grad():
        for i in range(frames):
            t = -span + 2 * span * i / (frames - 1)
            video.append(to_image(gen(base + t * d)[0]))
    video += video[::-1]  # walk out and back so it loops
    return _write_video(video, out, fps)


# --- latent directions ---------------------------------------------------------


def _extent(mask: torch.Tensor) -> torch.Tensor:
    """Length of the True-span along dim 1, per row."""
    idx = torch.arange(mask.size(1), device=mask.device)
    lo = torch.where(mask, idx, torch.full_like(idx, mask.size(1))).min(dim=1).values
    hi = torch.where(mask, idx, torch.full_like(idx, -1)).max(dim=1).values
    return (hi - lo + 1).clamp(min=1).float()


def measure_ink(x: torch.Tensor) -> torch.Tensor:
    """Ink coverage: a measurable proxy for a face's weight."""
    return (x > 0).float().mean(dim=(1, 2, 3))


def measure_extent(x: torch.Tensor) -> torch.Tensor:
    """Ink bounding-box width over height: a proxy for a face's width."""
    m = x > 0
    return _extent(m.any(dim=2).squeeze(1)) / _extent(m.any(dim=3).squeeze(1))


def find_direction(
    gen: nn.Module,
    measure: Measure,
    n: int = 2048,
    batch: int = 64,
    device: torch.device | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, float]:
    """Find the latent direction along which `measure` increases.

    A GAN has no encoder, so a semantic axis cannot be read off the model
    directly. Instead sample the latent space, measure the property on each
    output, and least-squares regress the property against the latent: the
    fitted coefficients point the way the property grows. Walking that vector
    traverses one typographic property deliberately, rather than drifting between
    arbitrary points the way a random interpolation does.

    Returns the unit direction and the correlation it achieves - a low
    correlation means the model never learned that property, usually because the
    dataset normalised it away.
    """
    device = device or pick_device()
    gen = gen.to(device)
    latent = _latent_of(gen)
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    with _sampling(gen), torch.no_grad():
        for _ in range(0, n, batch):
            z = torch.randn(batch, latent, generator=g).to(device)
            zs.append(z.cpu())
            ys.append(measure(gen(z)).cpu())
    z = torch.cat(zs).double()
    y = torch.cat(ys).double()

    zc = z - z.mean(0)
    yc = (y - y.mean()) / (y.std() + 1e-12)

    # Fit on one half and score on the other. With LATENT (200) coefficients an
    # in-sample fit is close to saturated at any sample count worth waiting for,
    # and would report a confident correlation for a direction that is noise.
    cut = len(zc) // 2
    d = torch.linalg.lstsq(zc[:cut], yc[:cut].unsqueeze(1)).solution.squeeze(1)
    d = d / d.norm()
    pred = zc[cut:] @ d
    held = yc[cut:]
    corr = float(
        ((pred - pred.mean()) * (held - held.mean())).mean() / (pred.std() * held.std() + 1e-12)
    )
    return d.float(), corr
