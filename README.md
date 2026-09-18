# GlyphGAN

<img src="./thumbnail.jpg" alt="Thumbnail" style="width: 300px; display: block;">

GlyphGAN trains a DCGAN on rasterised glyphs and renders a video that walks
through the latent space between them. The interpolation is the output: one
letterform morphing continuously into another through shapes that sit between
real typefaces.

## How it works

A DCGAN pairs a generator, which turns a random latent vector into an image,
against a discriminator that tries to tell generated glyphs from real ones.
Training the two against each other leaves the generator with a continuous
space of letterforms that can be walked through and sampled.

The model lives in `glyphgan.py`. `glyph-gan.ipynb` is a thin driver over it.

## Dataset

Any folder of images laid out for `torchvision.ImageFolder` works — one
subdirectory per class, images inside.

To build one from the fonts installed on a Mac, use
[fontscrape](https://github.com/latentcollection/macOS-fontface-scraper):

```sh
fontscrape --glyphs a --size 64 --mode xheight --per-family 3 --manifest --out dataset/
```

`--glyphs` writes one subdirectory per character. `--per-family 3` caps how many
weights each family contributes, which stops whichever families you happen to
own the most of from pulling the latent space toward themselves.

A single letter gives a model of that letter's design space, which interpolates
cleanly. Training on all 26 unconditioned averages them into mush.

## Usage

```sh
pip install -r requirements.txt
```

Then either run the notebook, or drive the module directly:

```python
import glyphgan as gg

gg.train("dataset/", size=64, epochs=50, batch_size=32)
gg.render_interpolation("checkpoints/generator.pt", "render.mp4", keys=8, frames=60)
```

Training checkpoints to `checkpoints/` every few epochs and on exit, including
Ctrl-C, and resumes from the latest checkpoint automatically. Rendering reads a
checkpoint from disk and needs nothing from the training session, so a video can
be made at any point from any run.

## Hyperparameters

| | Default | |
|---|---|---|
| `size` | `64` | output resolution; a power of two. Layer count follows it |
| `width` | `1024` | channels at the 4×4 stage, halving outward |
| `epochs` | `50` | |
| `batch_size` | `32` | |
| `lr` | `2e-4` | from the DCGAN paper |
| `betas` | `(0.5, 0.999)` | likewise |
| `LATENT` | `200` | latent vector size |

Start at 64px. A crisp small model that interpolates smoothly is more useful
than a soft large one, and a few thousand glyphs will not support a 256px
discriminator without memorising them.

## Output

Training prints losses and writes checkpoints. `render_interpolation` writes an
mp4 that travels through a sequence of latent waypoints and loops back to the
first. Interpolation is spherical rather than linear — a straight line between
two Gaussian latents passes through norms the model never saw during training,
which makes morphs sag and wash out halfway.

## Credits

Architecture after Radford et al., *Unsupervised Representation Learning with
Deep Convolutional Generative Adversarial Networks* (2015). Implementation
derived from the [PyTorch DCGAN tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
(BSD-3-Clause). Label smoothing and label flipping follow
[ganhacks](https://github.com/soumith/ganhacks).

Originally inspired by [Ritchie Vink's post on GANs and the distribution of
art](https://www.ritchievink.com/blog/2018/07/16/generative-adversarial-networks-in-pytorch-the-distribution-of-art/).

## License

MIT, see [LICENSE](LICENSE). This covers the code and nothing it reads or
produces — glyphs rendered from fonts you have licensed but do not own are a
separate question, and `fontscrape --license OFL` narrows a dataset to faces
that declare permissive terms.
