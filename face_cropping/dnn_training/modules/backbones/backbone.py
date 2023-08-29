from abc import ABC, abstractmethod

import torch.nn as nn


class Backbone(nn.Module, ABC):
    @abstractmethod
    def output_channels(self):
        raise NotImplementedError()

    @abstractmethod
    def output_strides(self):
        raise NotImplementedError()
