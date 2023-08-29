import os
import xml.etree.ElementTree as ET

from PIL import Image

from torch.utils.data import Dataset

from utils.path import to_path

CLASS_COUNT = 1000


class ClassificationImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        root = to_path(root)
        train_path = root / 'Data' / 'CLS-LOC' / 'train'
        self._class_names = [o.name for o in train_path.iterdir() if (train_path / o).is_dir()]
        self._class_names.sort()
        self._class_index_by_class_name = {c: i for i, c in enumerate(self._class_names)}

        if train:
            self._images = self._list_train_images(train_path)
        else:
            self._images = self._list_validation_images(root)

        self._transform = transform

    def _list_train_images(self, train_path):
        paths = []

        for i, class_name in enumerate(self._class_names):
            class_path = train_path / class_name

            for image in os.listdir(class_path):
                if _is_jpeg(image):
                    paths.append({'path': class_path / image, 'class_index': i})

        return paths

    def _list_validation_images(self, root):
        validation_path = root / 'Data' / 'CLS-LOC' / 'val'
        annotation_path = root / 'Annotations' / 'CLS-LOC' / 'val'
        paths = []

        for image in validation_path.iterdir():
            if _is_jpeg(image.name):
                xml_file = annotation_path / (image.stem + '.xml')
                paths.append({'path': validation_path / image, 'class_index': self._read_class(xml_file)})

        return paths

    def _read_class(self, xml_file):
        tree = ET.parse(xml_file)
        class_name = tree.getroot().findall("./object/name")[0].text
        return self._class_index_by_class_name[class_name]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = Image.open(self._images[index]['path']).convert('RGB')
        if self._transform is not None:
            image = self._transform(image)

        return image, self._images[index]['class_index']


def _is_jpeg(filename):
    filename = filename.upper()
    return filename.endswith('.JPEG') or filename.endswith('.JPG')
