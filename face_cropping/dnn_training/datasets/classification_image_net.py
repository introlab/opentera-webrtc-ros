import os
import xml.etree.ElementTree as ET

from PIL import Image

from torch.utils.data import Dataset

CLASS_COUNT = 1000


class ClassificationImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        train_path = os.path.join(root, 'Data/CLS-LOC/train')
        self._class_names = [o for o in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, o))]
        self._class_names.sort()
        self._class_index_by_class_name = {c: i for i, c in enumerate(self._class_names)}

        if train:
            self._images = self._list_train_images(train_path)
        else:
            self._images = self._list_validation_images(root)

        self._transform = transform

    def _list_train_images(self, train_path):
        paths = []

        for i in range(len(self._class_names)):
            class_path = os.path.join(train_path, self._class_names[i])

            for image in os.listdir(class_path):
                if _is_jpeg(image):
                    paths.append({'path': os.path.join(class_path, image), 'class_index': i})

        return paths

    def _list_validation_images(self, root):
        validation_path = os.path.join(root, 'Data/CLS-LOC/val')
        annotation_path = os.path.join(root, 'Annotations/CLS-LOC/val')
        paths = []

        for image in os.listdir(validation_path):
            if _is_jpeg(image):
                xml_file = os.path.join(annotation_path, os.path.splitext(image)[0] + '.xml')
                paths.append({'path': os.path.join(validation_path, image), 'class_index': self._read_class(xml_file)})

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
