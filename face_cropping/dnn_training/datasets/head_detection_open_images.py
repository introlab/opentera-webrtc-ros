import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from torch.utils.data import Dataset

from utils.path import to_path

HEAD_CLASS_ID = '/m/04hgtk'


class HeadDetectionOpenImages(Dataset):
    def __init__(self, root, split='training', transform=None, min_head_face_ratio=0.25, max_head_face_ratio=0.75):
        root = to_path(root)

        if split == 'training':
            self._root = root / 'train'
        elif split == 'validation':
            self._root = root / 'validation'
        elif split == 'testing':
            self._root = root / 'test'
        else:
            raise ValueError('Invalid split')

        self._transform = transform

        self._rotation_by_image_id = self._list_rotations()
        self._images = self._list_images(min_head_face_ratio, max_head_face_ratio)

    def _list_rotations(self):
        rotation_by_image_id = {}

        with open(self._root / 'metadata' / 'image_ids.csv', newline='') as image_id_file:
            image_id_reader = csv.reader(image_id_file, delimiter=',', quotechar='"')
            next(image_id_reader)
            for row in image_id_reader:
                image_id = row[0]
                rotation = 0.0 if row[11] == '' else float(row[11])

                rotation_by_image_id[image_id] = rotation

        return rotation_by_image_id

    def _list_images(self, min_head_face_ratio, max_head_face_ratio):
        bboxes = defaultdict(list)

        with open(self._root / 'labels' / 'detections.csv', newline='') as detection_file:
            detection_reader = csv.reader(detection_file, delimiter=',', quotechar='"')
            next(detection_reader)

            for row in detection_reader:
                image_id = row[0]
                class_id = row[2]
                x_min = float(row[4])
                x_max = float(row[5])
                y_min = float(row[6])
                y_max = float(row[7])

                width = x_max - x_min
                height = y_max - y_min

                if (class_id != HEAD_CLASS_ID or
                        width < min_head_face_ratio or
                        width > max_head_face_ratio or
                        height < min_head_face_ratio or
                        height > max_head_face_ratio):
                    continue

                bboxes[image_id].append([x_min, y_min, x_max, y_max])

        return self._convert_bboxes(bboxes)

    def _convert_bboxes(self, all_bboxes):
        images = []

        for image_id, bboxes in all_bboxes.items():
            path = Path('data') / f'{image_id}.jpg'
            if not self._is_valid_image_path(self._root / path):
                continue

            images.append({
                'path': path,
                'rotation': self._rotation_by_image_id[image_id],
                'bboxes': torch.tensor(bboxes)
            })

        return images

    @staticmethod
    def _is_valid_image_path(path):
        try:
            _ = Image.open(path).verify()
            return True
        except:
            return False

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._root / self._images[index]['path']).convert('RGB')
        bboxes = self._images[index]['bboxes'].clone()

        image = self._rotate_image(image, self._images[index]['rotation'])
        bboxes = self._rotate_bboxes(bboxes, self._images[index]['rotation'])

        initial_width, initial_height = image.size
        bboxes = self._scale_bboxes(bboxes, initial_width, initial_height)

        if self._transform is not None:
            image, target = self._transform(image, bboxes)

        return image, bboxes

    @staticmethod
    def _rotate_image(image, rotation):
        return image.rotate(rotation, expand=True)

    @staticmethod
    def _rotate_bboxes(bboxes, rotation):
        if rotation == 0.0:
            return bboxes
        elif rotation == 90.0:
            return torch.stack([bboxes[:, 1],
                                1.0 - bboxes[:, 2],
                                bboxes[:, 3],
                                1.0 - bboxes[:, 0]], dim=1)
        elif rotation == 180.0:
            return torch.stack([1.0 - bboxes[:, 2],
                                1.0 - bboxes[:, 3],
                                1.0 - bboxes[:, 0],
                                1.0 - bboxes[:, 1]], dim=1)
        elif rotation == 270.0:
            return torch.stack([1.0 - bboxes[:, 3],
                                bboxes[:, 0],
                                1.0 - bboxes[:, 1],
                                bboxes[:, 2]], dim=1)
        else:
            raise ValueError('Invalid rotation')

    @staticmethod
    def _scale_bboxes(bboxes, initial_width, initial_height):
        return torch.stack([bboxes[:, 0] * initial_width,
                            bboxes[:, 1] * initial_height,
                            bboxes[:, 2] * initial_width,
                            bboxes[:, 3] * initial_height], dim=1)
