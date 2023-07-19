import os

import torch
import torch.utils.data

from tqdm import tqdm

from .trainer import Trainer
from datasets import FaceDetectionWider, HeadDetectionOpenImages
from datasets import detection_collate, DetectionTrainingTransform, DetectionValidationTransform
from metrics import AveragePrecisionMetric, LossMetric, LossLearningCurves


class DetectorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', dataset_type='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128,
                 image_size=(224, 224),
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._image_size = image_size
        super(DetectorTrainer, self).__init__(device, model,
                                              dataset_root=dataset_root,
                                              output_path=output_path,
                                              epoch_count=epoch_count,
                                              learning_rate=learning_rate,
                                              weight_decay=weight_decay,
                                              batch_size=batch_size,
                                              batch_size_division=1,
                                              model_checkpoint=model_checkpoint)

        self._dataset_root = dataset_root

        self._learning_curves = LossLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()

    def _create_criterion(self, model):
        return model.loss

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = DetectionTrainingTransform(self._image_size)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'training', transform)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = DetectionValidationTransform(self._image_size)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'validation', transform)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        transform = DetectionValidationTransform(self._image_size)
        return self._create_dataset_loader(dataset_root, batch_size, batch_size_division, 'testing', transform)

    def _create_dataset_loader(self, dataset_root, batch_size, batch_size_division, split, transform):
        if self._dataset_type == 'wider_face':
            dataset = FaceDetectionWider(dataset_root,
                                         split=split if split != 'testing' else 'validation',
                                         transform=transform)
        elif self._dataset_type == 'open_images_head':
            dataset = HeadDetectionOpenImages(dataset_root, split=split, transform=transform)
        else:
            raise ValueError('Invalid dataset type')

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size // batch_size_division,
                                           shuffle=split == 'training',
                                           num_workers=8,
                                           collate_fn=detection_collate)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()

    def _move_target_to_device(self, targets, device):
        return [t.to(device) for t in targets]

    def _measure_training_metrics(self, loss, model_output, targets):
        self._training_loss_metric.add(loss.item())

        predictions, priors = model_output
        bboxes = self.model().decode_predictions(predictions, priors)

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, targets):
        self._validation_loss_metric.add(loss.item())

        predictions, priors = model_output
        bboxes = self.model().decode_predictions(predictions, priors)

    def _print_performances(self):
        print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
        print('Validation : Loss={}\n'.format(self._validation_loss_metric.get_loss()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        print('Evaluation - Detection', flush=True)
        ap50_metric = AveragePrecisionMetric(iou_threshold=0.5, confidence_threshold=0.01)
        ap75_metric = AveragePrecisionMetric(iou_threshold=0.75, confidence_threshold=0.01)
        ap90_metric = AveragePrecisionMetric(iou_threshold=0.90, confidence_threshold=0.01)

        for data in tqdm(dataset_loader):
            predictions, priors = model(data[0].to(device))
            targets = self._move_target_to_device(data[1], device)

            bboxes = model.decode_predictions(predictions, priors)
            ap50_metric.add(bboxes, targets)
            ap75_metric.add(bboxes, targets)
            ap90_metric.add(bboxes, targets)

        print('\nTest : Top 1 AP@0.5={}, AP@0.75={}, AP@0.9'.format(ap50_metric.get_value(),
                                                                    ap75_metric.get_value(),
                                                                    ap90_metric.get_value()))
