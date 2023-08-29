import numpy as np
from PIL import ImageOps, ImageEnhance

import torch.nn as nn


class RandomSharpnessChange(nn.Module):
    def __init__(self, sharpness_factor_range=(0.5, 2.0), p=0.5):
        super(RandomSharpnessChange, self).__init__()
        self._sharpness_factor_range = sharpness_factor_range
        self._p = p

    def forward(self, pil_image):
        if np.random.rand(1)[0] < self._p:
            sharpness_factor = np.random.uniform(self._sharpness_factor_range[0], self._sharpness_factor_range[1], 1)[0]
            enhancer = ImageEnhance.Sharpness(pil_image)
            return enhancer.enhance(sharpness_factor)
        else:
            return pil_image


class RandomAutocontrast(nn.Module):
    def __init__(self, p=0.5):
        super(RandomAutocontrast, self).__init__()
        self._p = p

    def forward(self, pil_image):
        if np.random.rand(1)[0] < self._p:
            return ImageOps.autocontrast(pil_image)
        else:
            return pil_image


class RandomEqualize(nn.Module):
    def __init__(self, p=0.5):
        super(RandomEqualize, self).__init__()
        self._p = p

    def forward(self, pil_image):
        if np.random.rand(1)[0] < self._p:
            return ImageOps.equalize(pil_image)
        else:
            return pil_image


class RandomPosterize(nn.Module):
    def __init__(self, bits_range=(2, 7), p=0.5):
        super(RandomPosterize, self).__init__()
        self._bits_range = bits_range
        self._p = p

    def forward(self, pil_image):
        if np.random.rand(1)[0] < self._p:
            bits = np.random.randint(self._bits_range[0], self._bits_range[1] + 1)
            return ImageOps.posterize(pil_image, bits)
        else:
            return pil_image
