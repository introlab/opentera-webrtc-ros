import torch
import torch.nn as nn
import torch.utils.data

import torchvision.transforms as transforms

from tqdm import tqdm

from .trainer import Trainer
from metrics import ClassificationAccuracyMetric, LossMetric, LossAccuracyLearningCurves
from metrics import TopNClassificationAccuracyMetric

from datasets import ClassificationImageNet
from datasets import RandomSharpnessChange, RandomAutocontrast, RandomEqualize, RandomPosterize


class BackboneTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128,
                 image_size=(224, 224),
                 model_checkpoint=None):
        self._image_size = image_size
        super(BackboneTrainer, self).__init__(device, model,
                                              dataset_root=dataset_root,
                                              output_path=output_path,
                                              epoch_count=epoch_count,
                                              learning_rate=learning_rate,
                                              weight_decay=weight_decay,
                                              batch_size=batch_size,
                                              batch_size_division=1,
                                              model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._learning_curves = LossAccuracyLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, model):
        return nn.CrossEntropyLoss()

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = create_training_image_transform(self._image_size)
        return create_dataset_loader(dataset_root, batch_size, batch_size_division, True, transform)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = create_validation_image_transform(self._image_size)
        return create_dataset_loader(dataset_root, batch_size, batch_size_division, False, transform)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, model_output, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_metric.add(model_output, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, target):
        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_metric.add(model_output, target)

    def _print_performances(self):
        print(f'\nTraining : Loss={self._training_loss_metric.get_loss()}, '
              f'Accuracy={self._training_accuracy_metric.get_accuracy()}')
        print(f'Validation : Loss={self._validation_loss_metric.get_loss()}, '
              f'Accuracy={self._validation_accuracy_metric.get_accuracy()}\n')

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(self._output_path / 'learning_curves.png',
                                   self._output_path / 'learning_curves.json')

    def _evaluate(self, model, device, dataset_loader, output_path):
        evaluate(model, device, dataset_loader)


def create_training_image_transform(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomSharpnessChange(),
        RandomAutocontrast(),
        RandomEqualize(),
        RandomPosterize(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing()
    ])


def create_validation_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def create_dataset_loader(dataset_root, batch_size, batch_size_division, train, transform):
    dataset = ClassificationImageNet(dataset_root, train=train, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size // batch_size_division, shuffle=train,
                                       num_workers=8)


def evaluate(model, device, dataset_loader):
    print('Evaluation - Classification', flush=True)
    top1_accuracy_metric = ClassificationAccuracyMetric()
    top5_accuracy_metric = TopNClassificationAccuracyMetric(5)

    for data in tqdm(dataset_loader):
        model_output = model(data[0].to(device))
        target = data[1].to(device)
        top1_accuracy_metric.add(model_output, target)
        top5_accuracy_metric.add(model_output, target)

    print(f'\nTest : Top 1 Accuracy={top1_accuracy_metric.get_accuracy()}, '
          f'Top 5 Accuracy={top5_accuracy_metric.get_accuracy()}')
