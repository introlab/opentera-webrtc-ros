import os

import torch
from PIL import Image

from torch.utils.data import Dataset


class FaceDetectionWider(Dataset):
    def __init__(self, root, split='training', transform=None):
        if split == 'training':
            self._images = self._list_images(root,
                                             'WIDER_train',
                                             os.path.join(root, 'wider_face_split', 'wider_face_train_bbx_gt.txt'))
        elif split == 'validation':
            self._images = self._list_images(root,
                                             'WIDER_val',
                                             os.path.join(root, 'wider_face_split', 'wider_face_val_bbx_gt.txt'))
        else:
            raise ValueError('Invalid split')

        self._transform = transform

    def _list_images(self, root, image_folder, annotation_file):
        images = []

        with open(annotation_file) as f:
            while True:
                image_path = f.readline().strip().replace('/', os.path.sep)
                if image_path == '':
                    break

                bbox_count = int(f.readline().strip())
                if bbox_count == 0:
                    f.readline()
                    continue

                bboxes = []
                for _ in range(bbox_count):
                    values = f.readline().strip().split(' ')
                    tl_x = float(values[0])
                    tl_y = float(values[1])
                    br_x = tl_x + float(values[2])
                    br_y = tl_y + float(values[3])
                    bboxes.append([tl_x, tl_y, br_x, br_y])

                images.append({
                    'path': os.path.join(root, image_folder, 'images', image_path),
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


# TODO remove after mosaic
if __name__ == '__main__':
    from PIL import ImageDraw

    d = FaceDetectionWider('/home/marc-antoine/Bureau/Maitrîse/dataset_data/WIDER')

    image, bboxes = d[4]

    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        draw.rectangle(bbox.tolist(), outline=(255, 0, 0))

    print(bboxes)

    image.show()


