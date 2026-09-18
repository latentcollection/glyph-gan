# Glossary

The domain here is typography, so typographic terms keep their typographic
meanings and the network borrows other words when it needs them. Two words in
particular are reserved: `width` and `axis` describe faces, never the model.

## Typography

**Glyph**:
One drawn shape from one face — the `a` of Helvetica Neue Bold. The unit of the
dataset: one image, one glyph.
_Avoid_: letter, character (a character is what a glyph depicts, not the shape)

**Character**:
The abstract symbol a glyph depicts, identified by codepoint. A face maps
characters to glyphs; a face with no glyph for a character is skipped, not
substituted.
_Avoid_: letter, glyph

**Face**:
One concrete typeface instance — Helvetica Neue Bold Italic. What a dataset
sample comes from, named by its PostScript name.
_Avoid_: font (ambiguous between face, family and file), style

**Family**:
A group of faces sharing a design — Helvetica Neue. Capping samples per family
keeps the model from leaning toward whichever families are most installed.
_Avoid_: typeface, font family

**Weight**:
How heavy a face's strokes are, light through black. A property of the face,
recorded by the scraper as a normalised trait.
_Avoid_: boldness, thickness

**Width**:
How condensed or extended a face is. A property of the face, independent of
weight. **Never** the network's channel count.
_Avoid_: aspect (that is a measured ratio, not the design property)

**Axis**:
A continuous design dimension a variable face exposes, such as `wght` or
`wdth`. **Never** a direction in latent space.
_Avoid_: dimension, direction

**X-height**:
The height of a lowercase `a` without ascenders. Normalising on it lets width
and weight survive as signal, where fitting each glyph to the frame flattens
them.

**Tofu**:
The `.notdef` placeholder box a face renders for a character it has no glyph
for. Not a letterform, and never wanted in a dataset.

## The model

**Latent vector**:
The random input a generator turns into a glyph image. Written `z`.
_Avoid_: noise, seed, embedding

**Latent direction**:
A vector in latent space along which one measured property increases. Found by
regression, not read off the model. **Never** called an axis.
_Avoid_: axis, dimension, feature

**Channels**:
The network's feature count at a given stage. The word `width` is reserved for
the typographic property.
_Avoid_: width, filters

**Ink coverage**:
The fraction of a rendered image that is ink. A measurable proxy for weight,
not weight itself.
_Avoid_: weight, density

**Extent**:
The width-to-height ratio of a glyph's ink bounding box. A measurable proxy for
width, not width itself.
_Avoid_: width, aspect ratio

**Size**:
Raster resolution in pixels, matching the scraper's `--size`. Typographic size
is always spelled *point size*, never bare "size".

**Generator checkpoint**:
The averaged generator weights alone, enough to render from and nothing more.
Distinct from a training checkpoint, which also carries the discriminator and
both optimiser states so a run can resume.
