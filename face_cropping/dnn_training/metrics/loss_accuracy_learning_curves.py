import json

import matplotlib.pyplot as plt
import numpy as np


class LossAccuracyLearningCurves:
    def __init__(self):
        self._training_loss_values = []
        self._validation_loss_values = []
        self._training_accuracy_values = []
        self._validation_accuracy_values = []

    def clear(self):
        self._training_loss_values = []
        self._validation_loss_values = []
        self._training_accuracy_values = []
        self._validation_accuracy_values = []

    def add_training_loss_value(self, value):
        self._training_loss_values.append(value)

    def add_validation_loss_value(self, value):
        self._validation_loss_values.append(value)

    def add_training_accuracy_value(self, value):
        self._training_accuracy_values.append(value)

    def add_validation_accuracy_value(self, value):
        self._validation_accuracy_values.append(value)

    def save(self, figure_path, data_path):
        self._save_figure(figure_path)
        self._save_data(data_path)

    def _save_figure(self, path):
        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        epochs = np.arange(1, len(self._training_loss_values) + 1)
        ax1.plot(epochs, self._training_loss_values, '-o', color='tab:blue', label='Training')
        epochs = range(1, len(self._validation_loss_values) + 1)
        ax1.plot(epochs, self._validation_loss_values, '-o', color='tab:orange', label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        epochs = np.arange(1, len(self._training_accuracy_values) + 1)
        ax2.plot(epochs, self._training_accuracy_values, '-o', color='tab:blue', label='Training')
        epochs = np.arange(1, len(self._validation_accuracy_values) + 1)
        ax2.plot(epochs, self._validation_accuracy_values, '-o', color='tab:orange', label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        fig.savefig(path)
        plt.close(fig)

    def _save_data(self, path):
        with open(path, 'w') as file:
            data = {
                'training_loss_values': self._training_loss_values,
                'validation_loss_values': self._validation_loss_values,
                'training_accuracy_values': self._training_accuracy_values,
                'validation_accuracy_values': self._validation_accuracy_values
            }
            json.dump(data, file, indent=4, sort_keys=True)
