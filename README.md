# GylphGAN

Deep convolutional GAN trained on glyphs

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
