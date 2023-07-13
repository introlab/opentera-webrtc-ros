import os

import torch

from tqdm import tqdm

from .trainer import Trainer
from datasets import OpenImagesHeadDetectorDataset
from datasets import HeadDetectorDatasetTrainingTransforms, HeadDetectorDatasetValidationTransforms
from metrics import LossMetric, AccuracyIoUMetric, LossAccuracyIouLearningCurves
from criterions import HeadDetectorLoss
from models import IMAGE_SIZE


class HeadDetectorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128, batch_size_division=1,
                 heatmap_sigma=10,
                 model_checkpoint=None):
        self._heatmap_sigma = heatmap_sigma

        super(HeadDetectorTrainer, self).__init__(device, model,
                                                  dataset_root=dataset_root,
                                                  output_path=output_path,
                                                  epoch_count=epoch_count,
                                                  learning_rate=learning_rate,
                                                  weight_decay=weight_decay,
                                                  batch_size=batch_size,
                                                  batch_size_division=batch_size_division,
                                                  model_checkpoint=model_checkpoint)

        self._training_loss_metric = LossMetric()
        self._training_accuracy_iou_metric = AccuracyIoUMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_accuracy_iou_metric = AccuracyIoUMetric()
        self._learning_curves = LossAccuracyIouLearningCurves()

    def _create_criterion(self, model):
        return HeadDetectorLoss()

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        dataset = OpenImagesHeadDetectorDataset(dataset_root,
                                                split='training',
                                                transforms=HeadDetectorDatasetTrainingTransforms(IMAGE_SIZE))
        return self._create_dataset_loader(dataset, batch_size, batch_size_division, shuffle=True)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        dataset = OpenImagesHeadDetectorDataset(dataset_root,
                                                split='validation',
                                                transforms=HeadDetectorDatasetValidationTransforms(IMAGE_SIZE))
        return self._create_dataset_loader(dataset, batch_size, batch_size_division)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        dataset = OpenImagesHeadDetectorDataset(dataset_root,
                                                split='testing',
                                                transforms=HeadDetectorDatasetValidationTransforms(IMAGE_SIZE))
        return self._create_dataset_loader(dataset, batch_size, batch_size_division)

    def _create_dataset_loader(self, dataset, batch_size, batch_size_division, shuffle=False):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=shuffle,
                                           num_workers=8)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()
        self._training_accuracy_iou_metric.clear()

    def _move_target_to_device(self, target, device):
        return target.to(device)

    def _measure_training_metrics(self, loss, prediction, target):
        self._training_loss_metric.add(loss.item())
        self._training_accuracy_iou_metric.add(prediction, target)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_accuracy_iou_metric.clear()

    def _measure_validation_metrics(self, loss, prediction, target):

        self._validation_loss_metric.add(loss.item())
        self._validation_accuracy_iou_metric.add(prediction, target)

    def _print_performances(self):
        print('\nTraining : Loss={}, Accuracy={}, IoU={}'.format(
            self._training_loss_metric.get_loss(),
            self._training_accuracy_iou_metric.get_accuracy(),
            self._training_accuracy_iou_metric.get_iou()))
        print('Validation : Loss={}, Accuracy={}, IoU={}\n'.format(
            self._validation_loss_metric.get_loss(),
            self._validation_accuracy_iou_metric.get_accuracy(),
            self._validation_accuracy_iou_metric.get_iou()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_iou_metric.get_accuracy())
        self._learning_curves.add_training_iou_value(self._training_accuracy_iou_metric.get_iou())

        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_iou_metric.get_accuracy())
        self._learning_curves.add_validation_iou_value(self._validation_accuracy_iou_metric.get_iou())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation', flush=True)
        accuracy_iou_metric = AccuracyIoUMetric()

        for data in tqdm(dataset_loader):
            model_output = model(data[0].to(device))
            target = self._move_target_to_device(data[1], device)
            accuracy_iou_metric.add(model_output, target)

        print('Testing : Accuracy={}, IoU={}\n'.format(
            accuracy_iou_metric.get_accuracy(),
            accuracy_iou_metric.get_iou()))
