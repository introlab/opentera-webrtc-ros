from pathlib import Path

import torch
from PIL import Image

from torch.utils.data import Dataset

from utils.path import to_path


class FaceDetectionWider(Dataset):
    def __init__(self, root, split='training', transform=None, min_head_face_ratio=0.1, max_head_face_ratio=1.0):
        root = to_path(root)
        if split == 'training':
            self._images = self._list_images(root,
                                             'WIDER_train',
                                             root / 'wider_face_split' / 'wider_face_train_bbx_gt.txt',
                                             min_head_face_ratio,
                                             max_head_face_ratio)
        elif split == 'validation':
            self._images = self._list_images(root,
                                             'WIDER_val',
                                             root / 'wider_face_split' / 'wider_face_val_bbx_gt.txt',
                                             min_head_face_ratio,
                                             max_head_face_ratio)
        else:
            raise ValueError('Invalid split')

        self._transform = transform

    @staticmethod
    def _list_images(root, image_folder, annotation_file, min_head_face_ratio, max_head_face_ratio):
        images = []

        with open(annotation_file) as f:
            while True:
                image_path = Path(f.readline().strip())
                if image_path == Path('.'):
                    break

                bbox_count = int(f.readline().strip())
                if bbox_count == 0:
                    f.readline()
                    continue

                full_image_path = root / image_folder / 'images' / image_path
                image = Image.open(full_image_path)

                bboxes = []
                for _ in range(bbox_count):
                    values = f.readline().strip().split(' ')
                    tl_x = float(values[0])
                    tl_y = float(values[1])
                    br_x = tl_x + float(values[2])
                    br_y = tl_y + float(values[3])

                    width_ratio = (br_x - tl_x) / image.width
                    height_ratio = (br_y - tl_y) / image.width

                    if (min_head_face_ratio <= width_ratio <= max_head_face_ratio and
                            min_head_face_ratio <= height_ratio <= max_head_face_ratio):
                        bboxes.append([tl_x, tl_y, br_x, br_y])

                if len(bboxes) > 0:
                    images.append({
                        'path': full_image_path,
                        'bboxes': torch.tensor(bboxes)
                    })

        return images

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]['path']).convert('RGB')
        bboxes = self._images[index]['bboxes'].clone()
        if self._transform is not None:
            image, bboxes = self._transform(image, bboxes)

        return image, bboxes
