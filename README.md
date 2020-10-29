# GylphGAN

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

Deep convolutional GAN trained on glyphs. Original architecture by [Ritchie Vink](https://www.ritchievink.com/blog/2018/07/16/generative-adversarial-networks-in-pytorch-the-distribution-of-art/), adapted by [
Aleksi Halttunen](https://github.com/aleksihalt/DCGAN_interpolation), and implemented by [Moritz Salla](https://github.com/moritzsalla).

![Thumbnail](./thumbnail.png)

Make sure to place your dataset in a subdir from root:

```python
data = torch.utils.data.DataLoader(ImageFolderEX(".", trans),
       batch_size=batch_size, shuffle=True,
       drop_last=True, num_workers=2)
```

```
.
└── 1
    ├── img1.jpg
    ├── img2.jpg
    └── img3.jpg
```
