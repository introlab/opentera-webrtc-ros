import os
import csv
import copy
from collections import defaultdict

import torch
from PIL import Image

from torch.utils.data import Dataset


HEAD_CLASS_IDS = {'/m/04hgtk', '/m/0dzct'}

TARGET_CONFIDENCE_INDEX = 0
TARGET_X_INDEX = 1
TARGET_Y_INDEX = 2
TARGET_W_INDEX = 3
TARGET_H_INDEX = 4


class OpenImagesHeadDetectorDataset(Dataset):
    def __init__(self, root, split=None, transforms=None):
        if split == 'training':
            self._root = os.path.join(root, 'train')
        elif split == 'validation':
            self._root = os.path.join(root, 'validation')
        elif split == 'testing':
            self._root = os.path.join(root, 'test')
        else:
            raise ValueError('Invalid split')

        self._transforms = transforms

        self._rotation_by_image_id = self._list_rotations()
        self._images, self._targets_by_image_id = self._list_images()

    def _list_rotations(self):
        rotation_by_image_id = {}

        with open(os.path.join(self._root, 'metadata', 'image_ids.csv'), newline='') as image_id_file:
            image_id_reader = csv.reader(image_id_file, delimiter=',', quotechar='"')
            next(image_id_reader)
            for row in image_id_reader:
                image_id = row[0]
                rotation = 0.0 if row[11] == '' else float(row[11])

                rotation_by_image_id[image_id] = rotation

        return rotation_by_image_id

    def _list_images(self):
        images_with_heads, targets_by_image_id_with_heads = self._list_images_with_heads()
        images_without_heads, targets_by_image_id_without_heads = self._list_images_without_heads(len(images_with_heads))

        images = images_with_heads + images_without_heads
        targets_by_image_id = targets_by_image_id_without_heads
        targets_by_image_id.update(targets_by_image_id_with_heads)

        return images, targets_by_image_id

    def _list_images_with_heads(self):
        image_ids = set()
        bboxes = defaultdict(list)

        with open(os.path.join(self._root, 'labels', 'detections.csv'), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                image_id = row[0]
                class_id = row[2]
                x_min = float(row[4])
                x_max = float(row[5])
                y_min = float(row[6])
                y_max = float(row[7])

                if class_id not in HEAD_CLASS_IDS:
                    continue

                image_ids.add(image_id)
                bboxes[image_id].append({
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                })

        images = self._image_ids_to_images(image_ids)
        return images, self._merge_heads(bboxes)

    def _list_images_without_heads(self, count):
        image_ids = set()
        image_ids_with_head = set()

        with open(os.path.join(self._root, 'labels', 'detections.csv'), newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                image_id = row[0]
                class_id = row[2]

                if class_id in HEAD_CLASS_IDS:
                    image_ids_with_head.add(image_id)
                image_ids.add(image_id)

        image_ids -= image_ids_with_head
        image_ids = list(image_ids)[:count]

        images = self._image_ids_to_images(image_ids)
        return images, {image_id:torch.zeros(5) for image_id in image_ids}

    def _image_ids_to_images(self, image_ids):
        image_ids = list(image_ids)
        image_ids.sort()

        images = []
        for i, image_id in enumerate(image_ids):
            path = os.path.join('data', '{}.jpg'.format(image_id))
            if not self._is_valid_image_path(os.path.join(self._root, path)):
                continue
            images.append({
                'image_id': image_id,
                'path': path,
                'rotation': self._rotation_by_image_id[image_id]
            })

        return images

    def _is_valid_image_path(self, path):
        try:
            _ = Image.open(path).verify()
            return True
        except:
            return False

    def _merge_heads(self, all_bboxes):
        merged_bboxes = {}
        for image_id, bboxes in all_bboxes.items():
            if len(bboxes) == 0:
                merged_bboxes[image_id] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
            else:
                x_min = min((b['x_min'] for b in bboxes))
                x_max = max((b['x_max'] for b in bboxes))
                y_min = min((b['y_min'] for b in bboxes))
                y_max = max((b['y_max'] for b in bboxes))
                x_center = (x_min + x_max) / 2.0
                y_center = (y_min + y_max) / 2.0
                w = x_max - x_min
                h = y_max - y_min

                merged_bboxes[image_id] = torch.tensor([1.0, x_center, y_center, w, h])

        return merged_bboxes

    def _rotate_image(self, image, rotation):
        return image.rotate(rotation, expand=True)

    def _rotate_target(self, target, rotation):
        if rotation == 0.0:
            return target
        elif rotation == 90.0:
            return torch.tensor([target[TARGET_CONFIDENCE_INDEX],
                                 target[TARGET_Y_INDEX],
                                 1.0 - target[TARGET_X_INDEX],
                                 target[TARGET_H_INDEX],
                                 target[TARGET_W_INDEX]])
        elif rotation == 180.0:
            return torch.tensor([target[TARGET_CONFIDENCE_INDEX],
                                 1.0 - target[TARGET_X_INDEX],
                                 1.0 - target[TARGET_Y_INDEX],
                                 target[TARGET_W_INDEX],
                                 target[TARGET_H_INDEX]])
        elif rotation == 270.0:
            return torch.tensor([target[TARGET_CONFIDENCE_INDEX],
                                 1.0 - target[TARGET_Y_INDEX],
                                 target[TARGET_X_INDEX],
                                 target[TARGET_H_INDEX],
                                 target[TARGET_W_INDEX]])
        else:
            raise ValueError('Invalid rotation')

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self._root, self._images[index]['path'])).convert('RGB')
        target = self._targets_by_image_id[self._images[index]['image_id']]
        target = self._load_target(target)

        image = self._rotate_image(image, self._images[index]['rotation'])
        target = self._rotate_target(target, self._images[index]['rotation'])

        initial_width, initial_height = image.size

        metadata = {
            'initial_width': initial_width,
            'initial_height': initial_height,
        }

        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target, metadata

    def _load_target(self, target):
        return copy.deepcopy(target)
