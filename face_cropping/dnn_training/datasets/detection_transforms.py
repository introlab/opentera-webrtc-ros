import random

from PIL import Image

import torch
import torch.utils.data
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from .transforms import RandomSharpnessChange, RandomAutocontrast, RandomEqualize, RandomPosterize


def detection_collate(batch):
    images = torch.utils.data.dataloader.default_collate([e[0] for e in batch])
    target = [e[1] for e in batch]

    return images, target


def resize_image(image, size):
    w, h = image.size
    scale = min(size[0] / h, size[1] / w)

    image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    padded_image = Image.new('RGB', (size[0], size[1]), (114, 114, 114))
    padded_image.paste(image, (0, 0))

    return padded_image, scale


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def _horizontal_flip(image, bboxes):
    flipped_image = F.hflip(image)
    w, h = flipped_image.size

    flipped_bboxes = torch.stack([w - bboxes[:, 2],
                                  bboxes[:, 1],
                                  w - bboxes[:, 0],
                                  bboxes[:, 3]], dim=1)

    return flipped_image, flipped_bboxes


class DetectionTrainingTransform(nn.Module):
    def __init__(self, image_size, horizontal_flip_p=0.5):
        super().__init__()
        self._image_size = image_size

        self._image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            RandomSharpnessChange(),
            RandomAutocontrast(),
            RandomEqualize(),
            RandomPosterize(),
        ])

        self._horizontal_flip_p = horizontal_flip_p

    def forward(self, image, bboxes):
        image = self._image_only_transform(image)

        resized_image, scale = resize_image(image, self._image_size)
        resized_bboxes = bboxes * scale

        if random.random() < self._horizontal_flip_p:
            resized_image, resized_bboxes = _horizontal_flip(resized_image, resized_bboxes)

        resized_image_tensor = F.to_tensor(resized_image)
        return normalize(resized_image_tensor), resized_bboxes


class DetectionValidationTransform(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self._image_size = image_size

    def forward(self, image, bboxes=None):
        resized_image, scale = resize_image(image, self._image_size)
        normalized_tensor = normalize(F.to_tensor(resized_image))
        if bboxes is None:
            return normalized_tensor, scale
        else:
            return normalized_tensor, bboxes * scale
