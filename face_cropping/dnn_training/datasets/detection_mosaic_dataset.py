import random

import torch
from PIL import Image

from torch.utils.data import Dataset

from datasets.detection_transforms import resize_image

BACKGROUND_COLOR = (114, 114, 114)


class DetectionMosaicDataset(Dataset):
    def __init__(self, dataset, image_size, transform=None, min_ratio=0.3, mosaic_p=0.5):
        self._dataset = dataset
        self._image_size = image_size

        self._min_ratio = min_ratio
        self._mosaic_p = mosaic_p

        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        if random.random() < self._mosaic_p:
            image, bboxes = self._get_item_mosaic(index)
        else:
            image, bboxes = self._get_item_normal(index)

        if self._transform is not None:
            image, bboxes = self._transform(image, bboxes)

        return image, bboxes

    def _get_item_mosaic(self, index):
        mosaic_image = Image.new('RGB', (self._image_size[1], self._image_size[0]), BACKGROUND_COLOR)

        image0_ratio = random.uniform(self._min_ratio, 1.0 - self._min_ratio)
        image0_max_size = (image0_ratio * mosaic_image.width, image0_ratio * mosaic_image.height)
        image0, bboxes0 = self._dataset[index]
        image0, scale0 = self._mosaic_resize_image(image0, image0_max_size)
        offset0 = (0, 0)
        mosaic_image.paste(image0, offset0)

        image1_max_size = (mosaic_image.width - image0.width, mosaic_image.height)
        image1, bboxes1 = self._dataset[random.randrange(0, len(self._dataset))]
        image1, scale1 = self._mosaic_resize_image(image1, image1_max_size)
        offset1 = (image0.width, 0)
        mosaic_image.paste(image1, offset1)

        bboxes01 = torch.cat([self._transform_bboxes(bboxes0, scale0, offset0),
                              self._transform_bboxes(bboxes1, scale1, offset1)], dim=0)

        image2_max_size = (mosaic_image.width, mosaic_image.height - min(image0.height, image1.height))
        image2, bboxes2 = self._dataset[random.randrange(0, len(self._dataset))]
        image2, scale2, offset2 = self._find_max_image(mosaic_image, image2, image2_max_size, self._bottom_left_offset)

        if not self._is_valid_image(mosaic_image, image2):
            return mosaic_image, bboxes01
        mosaic_image.paste(image2, offset2)

        bboxes012 = torch.cat([bboxes01, self._transform_bboxes(bboxes2, scale2, offset2)], dim=0)

        image3_max_size = (mosaic_image.width - min(image0.width, image2.width),
                           mosaic_image.height - min(image0.height, image1.height))
        image3, bboxes3 = self._dataset[random.randrange(0, len(self._dataset))]
        image3, scale3, offset3 = self._find_max_image(mosaic_image, image3, image3_max_size, self._bottom_right_offset)

        if not self._is_valid_image(mosaic_image, image3):
            return mosaic_image, bboxes012
        mosaic_image.paste(image3, offset3)

        return mosaic_image, torch.cat([bboxes012, self._transform_bboxes(bboxes3, scale3, offset3)], dim=0)

    @staticmethod
    def _get_scale_from_max_size(image, max_size):
        w, h = image.size
        scale = min(max_size[0] / w, max_size[1] / h)

        size = (int(w * scale), int(h * scale))
        if size[0] <= 0 or size[1] <= 0:
            return None, None

        return scale, size

    @staticmethod
    def _mosaic_resize_image(image, max_size):
        scale, size = DetectionMosaicDataset._get_scale_from_max_size(image, max_size)

        if scale is None:
            return None, None

        image = image.resize(size, Image.BILINEAR)
        return image, scale

    def _find_max_image(self, mosaic_image, image, image_max_size, offset_fn, scale_step_ratio=0.95):
        while True:
            _, size = self._get_scale_from_max_size(image, image_max_size)
            if size is None:
                break
            offset = offset_fn(mosaic_image, size)

            crop = mosaic_image.crop((offset[0], offset[1],
                                      offset[0] + size[0], offset[1] + size[1]))
            crop_extrema = crop.convert("L").getextrema()
            if crop_extrema[0] == crop_extrema[1]:
                resized_image, scale = self._mosaic_resize_image(image, image_max_size)
                return resized_image, scale, offset
            else:
                image_max_size = (int(image_max_size[0] * scale_step_ratio), int(image_max_size[1] * scale_step_ratio))

        return None, None, None

    @staticmethod
    def _bottom_left_offset(mosaic_image, size):
        return 0, mosaic_image.height - size[1]

    @staticmethod
    def _bottom_right_offset(mosaic_image, size):
        return mosaic_image.width - size[0], mosaic_image.height - size[1]

    def _is_valid_image(self, mosaic_image, image):
        return (image is not None and
                image.width / mosaic_image.width >= self._min_ratio and
                image.height / mosaic_image.height >= self._min_ratio)

    @staticmethod
    def _transform_bboxes(bboxes, scale, offset):
        bboxes = bboxes * scale
        bboxes[:, 0] += offset[0]
        bboxes[:, 1] += offset[1]
        bboxes[:, 2] += offset[0]
        bboxes[:, 3] += offset[1]
        return bboxes

    def _get_item_normal(self, index):
        image, bboxes = self._dataset[index]
        image, scale = resize_image(image, self._image_size)

        return image, bboxes
