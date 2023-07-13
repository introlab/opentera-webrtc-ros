import random

import numpy as np
from PIL import ImageOps, ImageEnhance

import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from .open_images_head_detector_dataset import TARGET_CLASS_INDEX, TARGET_X_INDEX, TARGET_Y_INDEX, TARGET_W_INDEX
from .open_images_head_detector_dataset import TARGET_H_INDEX


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


class HeadDetectorDatasetTrainingTransforms:
    def __init__(self, image_size):
        self._horizontal_flip_p = 0.5

        self._image_only_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            RandomSharpnessChange(),
            RandomAutocontrast(),
            RandomEqualize(),
            RandomPosterize(),
            transforms.ToTensor()
        ])

    def __call__(self, image, target):
        if random.random() < self._horizontal_flip_p:
            image = F.hflip(image)
            target = _hflip_target(target)

        image = self._image_only_transform(image)
        return image, target


def _hflip_target(target):
    return torch.tensor([target[TARGET_CLASS_INDEX],
                         1.0 - target[TARGET_X_INDEX],
                         target[TARGET_Y_INDEX],
                         target[TARGET_W_INDEX],
                         target[TARGET_H_INDEX]])


class HeadDetectorDatasetValidationTransforms:
    def __init__(self, image_size):
        self._image_only_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __call__(self, image, target):
        image = self._image_only_transform(image)
        return image, target
