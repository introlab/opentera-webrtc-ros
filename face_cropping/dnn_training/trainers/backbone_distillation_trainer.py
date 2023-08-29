from .distillation_trainer import DistillationTrainer
from criterions import DistillationCrossEntropyLoss
from metrics import ClassificationAccuracyMetric, LossMetric, LossAccuracyLearningCurves

from .backbone_trainer import create_training_image_transform, create_validation_image_transform, create_dataset_loader
from .backbone_trainer import evaluate


class BackboneDistillationTrainer(DistillationTrainer):
    def __init__(self, device, student_model, teacher_model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128,
                 image_size=(224, 224),
                 student_model_checkpoint=None, teacher_model_checkpoint=None, loss_alpha=0.25):
        self._image_size = image_size
        self._loss_alpha = loss_alpha

        super().__init__(device, student_model, teacher_model,
                         dataset_root=dataset_root,
                         output_path=output_path,
                         epoch_count=epoch_count,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         batch_size=batch_size,
                         batch_size_division=1,
                         student_model_checkpoint=student_model_checkpoint,
                         teacher_model_checkpoint=teacher_model_checkpoint)

        self._dataset_root = dataset_root

        self._learning_curves = LossAccuracyLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._training_accuracy_metric = ClassificationAccuracyMetric()
        self._validation_accuracy_metric = ClassificationAccuracyMetric()

    def _create_criterion(self, student_model, teacher_model):
        return DistillationCrossEntropyLoss(alpha=self._loss_alpha)

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
        print(f'Validation : Loss={self._validation_loss_metric.get_loss()},'
              f' Accuracy={self._validation_accuracy_metric.get_accuracy()}\n')

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_training_accuracy_value(self._training_accuracy_metric.get_accuracy())
        self._learning_curves.add_validation_accuracy_value(self._validation_accuracy_metric.get_accuracy())

        self._learning_curves.save(self._output_path / 'learning_curves.png',
                                   self._output_path / 'learning_curves.json')

    def _evaluate(self, model, device, dataset_loader, output_path):
        evaluate(model, device, dataset_loader)
