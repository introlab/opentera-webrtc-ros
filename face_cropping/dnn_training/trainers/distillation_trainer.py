from abc import ABC, abstractmethod
import os

import torch
import torch.nn as nn

from tqdm import tqdm

from modules import load_checkpoint
from utils.path import to_path


class DistillationTrainer(ABC):
    def __init__(self, device, student_model, teacher_model, dataset_root='', output_path='',
                 epoch_count=10, learning_rate=0.01, weight_decay=0.0, batch_size=128, batch_size_division=4,
                 student_model_checkpoint=None, teacher_model_checkpoint=None):
        self._device = device
        self._output_path = to_path(output_path)
        os.makedirs(self._output_path, exist_ok=True)

        self._epoch_count = epoch_count
        self._batch_size = batch_size
        self._batch_size_division = batch_size_division

        self._criterion = self._create_criterion(student_model, teacher_model)

        if teacher_model_checkpoint is None:
            raise ValueError('teacher_model_checkpoint should be set.')

        load_checkpoint(teacher_model, teacher_model_checkpoint, strict=True)
        if student_model_checkpoint is not None:
            load_checkpoint(student_model, student_model_checkpoint, strict=False)
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('DataParallel - GPU count:', torch.cuda.device_count())
            student_model = nn.DataParallel(student_model)
            teacher_model = nn.DataParallel(teacher_model)

        self._student_model = student_model.to(device)
        self._teacher_model = teacher_model.to(device)
        bias_parameters = [parameter for name, parameter in student_model.named_parameters() if name.endswith('.bias')]
        other_parameters = [parameter for name, parameter in student_model.named_parameters() if
                            not name.endswith('.bias')]
        parameter_groups = [
            {'params': other_parameters},
            {'params': bias_parameters, 'weight_decay': 0.0}
        ]
        self._optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, weight_decay=weight_decay)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, epoch_count)

        self._training_dataset_loader = self._create_training_dataset_loader(dataset_root,
                                                                             batch_size,
                                                                             batch_size_division)
        self._validation_dataset_loader = self._create_validation_dataset_loader(dataset_root,
                                                                                 batch_size,
                                                                                 batch_size_division)
        self._testing_dataset_loader = self._create_testing_dataset_loader(dataset_root,
                                                                           batch_size,
                                                                           batch_size_division)
        if self._testing_dataset_loader is None:
            self._testing_dataset_loader = self._validation_dataset_loader

    @abstractmethod
    def _create_criterion(self, student_model, teacher_model):
        raise NotImplementedError()

    @abstractmethod
    def _create_training_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        raise NotImplementedError()

    @abstractmethod
    def _create_validation_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        raise NotImplementedError()

    def _create_testing_dataset_loader(self, dataset_root, batch_size, batch_size_division):
        return None

    def train(self):
        self._teacher_model.eval()

        self._clear_between_training()
        for epoch in range(self._epoch_count):
            print(f'Training - Epoch [{epoch + 1}/{self._epoch_count}]', flush=True)
            self._train_one_epoch()

            print(f'\nValidation - Epoch [{epoch + 1}/{self._epoch_count}]', flush=True)
            self._validate()
            self._scheduler.step()
            next_epoch_method = getattr(self._criterion, 'next_epoch', None)
            if next_epoch_method is not None:
                next_epoch_method()

            self._print_performances()
            self._save_learning_curves()
            self._save_states(epoch + 1)

        with torch.no_grad():
            self._student_model.eval()
            self._evaluate(self._student_model, self._device, self._testing_dataset_loader, self._output_path)

    @abstractmethod
    def _clear_between_training(self):
        raise NotImplementedError()

    def _train_one_epoch(self):
        self._clear_between_training_epoch()

        self._student_model.train()
        self._optimizer.zero_grad()

        division = 0
        for data in tqdm(self._training_dataset_loader):
            input_data = data[0].to(self._device)
            with torch.no_grad():
                teacher_model_output = self._teacher_model(input_data)
            student_model_output = self._student_model(input_data)
            target = self._move_target_to_device(data[1], self._device)
            loss = self._criterion(student_model_output, target, teacher_model_output)

            if torch.all(torch.isfinite(loss)):
                loss.backward()
                self._measure_training_metrics(loss, student_model_output, target)
            else:
                print('Warning the loss is not finite.')

            division += 1
            if division == self._batch_size_division:
                division = 0
                torch.nn.utils.clip_grad_value_(self._student_model.parameters(), 1)
                self._optimizer.step()
                self._optimizer.zero_grad()

        if division != 0:
            torch.nn.utils.clip_grad_value_(self._student_model.parameters(), 1)
            self._optimizer.step()
            self._optimizer.zero_grad()

    @abstractmethod
    def _clear_between_training_epoch(self):
        raise NotImplementedError()

    @abstractmethod
    def _move_target_to_device(self, target, device):
        raise NotImplementedError()

    @abstractmethod
    def _measure_training_metrics(self, loss, model_output, target):
        raise NotImplementedError()

    def _validate(self):
        with torch.no_grad():
            self._clear_between_validation_epoch()

            self._student_model.eval()

            for data in tqdm(self._validation_dataset_loader):
                input_data = data[0].to(self._device)
                teacher_model_output = self._teacher_model(input_data)
                student_model_output = self._student_model(input_data)
                target = self._move_target_to_device(data[1], self._device)
                loss = self._criterion(student_model_output, target, teacher_model_output)

                if torch.all(torch.isfinite(loss)):
                    self._measure_validation_metrics(loss, student_model_output, target)
                else:
                    print('Warning the loss is not finite.')

    @abstractmethod
    def _clear_between_validation_epoch(self):
        raise NotImplementedError()

    @abstractmethod
    def _measure_validation_metrics(self, loss, model_output, target):
        raise NotImplementedError()

    @abstractmethod
    def _print_performances(self):
        raise NotImplementedError()

    @abstractmethod
    def _save_learning_curves(self):
        raise NotImplementedError()

    def _save_states(self, epoch):
        torch.save(self._student_model.state_dict(), self._output_path / f'model_checkpoint_epoch_{epoch}.pth')

    @abstractmethod
    def _evaluate(self, model, device, dataset_loader, output_path):
        raise NotImplementedError()

    def student_model(self):
        if isinstance(self._student_model, nn.DataParallel):
            return self._student_model.module
        else:
            return self._student_model
