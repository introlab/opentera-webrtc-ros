from .detector_trainer import create_training_dataset_loader, create_validation_dataset_loader
from .detector_trainer import create_testing_dataset_loader, move_target_to_device, evaluate
from .distillation_trainer import DistillationTrainer
from criterions import DistillationSingleClassDetectionLoss
from metrics import AveragePrecisionMetric, LossMetric, LossAveragePrecisionLearningCurves


class DetectorDistillationTrainer(DistillationTrainer):
    def __init__(self, device, student_model, teacher_model, dataset_root='', dataset_type='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128,
                 image_size=(224, 224), use_mosaic=False,
                 student_model_checkpoint=None, teacher_model_checkpoint=None, loss_alpha=0.25):
        self._dataset_type = dataset_type
        self._image_size = image_size
        self._use_mosaic = use_mosaic
        self._loss_alpha = loss_alpha
        super(DetectorDistillationTrainer, self).__init__(device, student_model, teacher_model,
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

        self._learning_curves = LossAveragePrecisionLearningCurves()

        self._training_loss_metric = LossMetric()
        self._validation_loss_metric = LossMetric()
        self._validation_ap_metric = AveragePrecisionMetric(iou_threshold=0.5, confidence_threshold=0.1)

    def _create_criterion(self, student_model, teacher_model):
        detection_loss = DistillationSingleClassDetectionLoss(alpha=self._loss_alpha)

        def loss(student_model_output, targets, teacher_model_output):
            student_predictions, student_priors = student_model_output
            student_decoded_bboxes = student_model.decode_predictions(student_predictions, student_priors)

            teacher_predictions, teacher_priors = teacher_model_output
            teacher_decoded_bboxes = teacher_model.decode_predictions(student_predictions, student_priors)

            return detection_loss(student_predictions, student_priors, student_decoded_bboxes,
                                  teacher_predictions, teacher_priors, teacher_decoded_bboxes,
                                  targets)

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
        decode_bboxes = self.student_model().decode_predictions(predictions, priors)
        self._validation_ap_metric.add(decode_bboxes, targets)

    def _print_performances(self):
        print(f'\nTraining : Loss={self._training_loss_metric.get_loss()}')
        print(f'Validation : Loss={self._validation_loss_metric.get_loss()}, '
              f'AP@0.5={self._validation_ap_metric.get_value()}\n')

    def _save_learning_curves(self):
        self._learning_curves.add_training_loss_value(self._training_loss_metric.get_loss())
        self._learning_curves.add_validation_loss_value(self._validation_loss_metric.get_loss())
        self._learning_curves.add_validation_ap_value(self._validation_ap_metric.get_value())

        self._learning_curves.save(self._output_path / 'learning_curves.png',
                                   self._output_path / 'learning_curves.json')

    def _evaluate(self, student_model, device, dataset_loader, output_path):
        evaluate(student_model, self.student_model(), device, dataset_loader, output_path,
                 self._dataset_type, self._dataset_root, self._image_size)
