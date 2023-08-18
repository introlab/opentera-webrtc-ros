import os

import torch
import torch.utils.data

from PIL import Image

from tqdm import tqdm

from .trainer import Trainer
from criterions import SingleClassDetectionLoss
from datasets import FaceDetectionWider, HeadDetectionOpenImages
from datasets import detection_collate, DetectionTrainingTransform, DetectionValidationTransform, DetectionMosaicDataset
from datasets import BeforeMosaicDetectionTrainingTransform, AfterMosaicDetectionTrainingTransform
from metrics import AveragePrecisionMetric, LossMetric, LossAveragePrecisionLearningCurves
from modules.heads import filter_decoded_bboxes, CONFIDENCE_INDEX, TL_X_INDEX, TL_Y_INDEX, BR_X_INDEX, BR_Y_INDEX


class DetectorTrainer(Trainer):
    def __init__(self, device, model, dataset_root='', dataset_type='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128,
                 image_size=(224, 224), use_mosaic=False,
                 model_checkpoint=None):
        self._dataset_type = dataset_type
        self._image_size = image_size
        self._use_mosaic = use_mosaic
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

        self._learning_curves = LossAveragePrecisionLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_ap_metric = AveragePrecisionMetric(iou_threshold=0.5, confidence_threshold=0.1)

    def _create_criterion(self, model):
        detection_loss = SingleClassDetectionLoss(confidence_loss_type='focal_loss', bbox_loss_type='eiou_loss')

        def loss(model_output, targets):
            predictions, priors = model_output
            decoded_bboxes = model.decode_predictions(predictions, priors)
            return detection_loss(predictions, priors, decoded_bboxes, targets)

        return loss

    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return create_training_dataset_loader(self._dataset_type, dataset_root,
                                              self._image_size, self._use_mosaic, batch_size, batch_size_division)

    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return create_validation_dataset_loader(self._dataset_type, dataset_root,
                                                self._image_size, batch_size, batch_size_division)

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return create_testing_dataset_loader(self._dataset_type, dataset_root,
                                             self._image_size, batch_size, batch_size_division)

    def _clear_between_training(self):
        self._learning_curves.clear()

    def _clear_between_training_epoch(self):
        self._training_loss_metric.clear()

    def _move_target_to_device(self, targets, device):
        return move_target_to_device(targets, device)

    def _measure_training_metrics(self, loss, model_output, targets):
        self._training_loss_metric.add(loss.item())

    def _clear_between_validation_epoch(self):
        self._validation_loss_metric.clear()
        self._validation_ap_metric.clear()

    def _measure_validation_metrics(self, loss, model_output, targets):
        self._validation_loss_metric.add(loss.item())

        predictions, priors = model_output
        decode_bboxes = self.model().decode_predictions(predictions, priors)
        self._validation_ap_metric.add(decode_bboxes, targets)

    def _print_performances(self):
        print('\nTraining : Loss={}'.format(self._training_loss_metric.get_loss()))
        print('Validation : Loss={}, AP@0.5={}\n'.format(self._validation_loss_metric.get_loss(),
                                                         self._validation_ap_metric.get_value()))

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_ap_value(self._validation_ap_metric.get_value())

        self._learning_curves.save(os.path.join(self._output_path, 'learning_curves.png'),
                                   os.path.join(self._output_path, 'learning_curves.json'))

    def _evaluate(self, model, device, dataset_loader, output_path):
        evaluate(model, self.model(), device, dataset_loader, output_path,
                 self._dataset_type, self._dataset_root, self._image_size)


def create_training_dataset_loader(dataset_type, dataset_root, image_size, use_mosaic, batch_size, batch_size_division):
    if use_mosaic:
        transform = BeforeMosaicDetectionTrainingTransform()
        dataset = _create_dataset(dataset_type, dataset_root, 'training', transform)

        transform = AfterMosaicDetectionTrainingTransform()
        dataset = DetectionMosaicDataset(dataset, image_size, transform=transform)
    else:
        transform = DetectionTrainingTransform(image_size)
        dataset = _create_dataset(dataset_type, dataset_root, 'training', transform)

    return _create_dataset_loader(dataset, batch_size, batch_size_division, shuffle=True)


def create_validation_dataset_loader(dataset_type, dataset_root, image_size, batch_size, batch_size_division):
    transform = DetectionValidationTransform(image_size)
    dataset = _create_dataset(dataset_type, dataset_root, 'validation', transform)
    return _create_dataset_loader(dataset, batch_size, batch_size_division, shuffle=False)


def create_testing_dataset_loader(dataset_type, dataset_root, image_size, batch_size, batch_size_division):
    transform = DetectionValidationTransform(image_size)
    dataset = _create_dataset(dataset_type, dataset_root, 'testing', transform)
    return _create_dataset_loader(dataset, batch_size, batch_size_division, shuffle=False)


def _create_dataset(dataset_type, dataset_root, split, transform):
    if dataset_type == 'wider_face':
        return FaceDetectionWider(dataset_root,
                                  split=split if split != 'testing' else 'validation',
                                  transform=transform)
    elif dataset_type == 'open_images_head':
        return HeadDetectionOpenImages(dataset_root, split=split, transform=transform)
    else:
        raise ValueError('Invalid dataset type')


def _create_dataset_loader(dataset, batch_size, batch_size_division, shuffle):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size // batch_size_division,
                                       shuffle=shuffle,
                                       num_workers=8,
                                       collate_fn=detection_collate)


def evaluate(model, single_gpu_model, device, dataset_loader, output_path, dataset_type, dataset_root, image_size):
    _evaluate_all_datasets(model, device, dataset_loader, output_path)

    if dataset_type == 'wider_face':
        _write_wider_face_val_results(dataset_root, image_size, single_gpu_model, device, output_path)


def _evaluate_all_datasets(model, device, dataset_loader, output_path):
    print('Evaluation - Detection', flush=True)

    ap25_metric = AveragePrecisionMetric(iou_threshold=0.25, confidence_threshold=0.01)
    ap50_metric = AveragePrecisionMetric(iou_threshold=0.5, confidence_threshold=0.01)
    ap75_metric = AveragePrecisionMetric(iou_threshold=0.75, confidence_threshold=0.01)
    ap90_metric = AveragePrecisionMetric(iou_threshold=0.90, confidence_threshold=0.01)

    for data in tqdm(dataset_loader):
        predictions, priors = model(data[0].to(device))
        targets = move_target_to_device(data[1], device)

        bboxes = model.decode_predictions(predictions, priors)
        ap25_metric.add(bboxes, targets)
        ap50_metric.add(bboxes, targets)
        ap75_metric.add(bboxes, targets)
        ap90_metric.add(bboxes, targets)

    print('\nTest : AP@25={}, AP@0.5={}, AP@0.75={}, AP@0.9={}'.format(ap25_metric.get_value(),
                                                                       ap50_metric.get_value(),
                                                                       ap75_metric.get_value(),
                                                                       ap90_metric.get_value()))
    ap25_metric.save_curve(output_path, suffix='_25')
    ap50_metric.save_curve(output_path, suffix='_50')
    ap75_metric.save_curve(output_path, suffix='_75')
    ap90_metric.save_curve(output_path, suffix='_90')


def _write_wider_face_val_results(dataset_root, image_size, model, device, output_path):
    print('Evaluation - WiderFace Results', flush=True)

    test_root_path = os.path.join(dataset_root, 'WIDER_val', 'images')
    output_root_path = os.path.join(output_path, 'wider_results')

    transform = DetectionValidationTransform(image_size)

    categories = os.listdir(test_root_path)

    for category in tqdm(categories):
        category_input_path = os.path.join(test_root_path, category)
        category_output_path = os.path.join(output_root_path, category)
        image_filenames = os.listdir(category_input_path)

        os.makedirs(category_output_path, exist_ok=True)

        for image_filename in image_filenames:
            image = Image.open(os.path.join(category_input_path, image_filename)).convert('RGB')

            input_tensor, scale = transform(image)
            predictions, priors = model(input_tensor.unsqueeze(0).to(device))
            bboxes = model.decode_predictions(predictions, priors)[0]
            bboxes = filter_decoded_bboxes(bboxes, confidence_threshold=0.01, nms_threshold=0.45)

            with open(os.path.join(category_output_path, os.path.splitext(image_filename)[0] + '.txt'), 'w') as f:
                f.write(image_filename + '\n')
                f.write(f'{bboxes.size(0)}\n')

                for bbox in bboxes:
                    c = bbox[CONFIDENCE_INDEX].item()
                    tl_x = round(bbox[TL_X_INDEX].item() / scale)
                    tl_y = round(bbox[TL_Y_INDEX].item() / scale)
                    br_x = round(bbox[BR_X_INDEX].item() / scale)
                    br_y = round(bbox[BR_Y_INDEX].item() / scale)
                    f.write(f'{tl_x} {tl_y} {br_x - tl_x} {br_y - tl_y} {c}\n')


def move_target_to_device(targets, device):
    return [t.to(device) for t in targets]
