import torch.nn as nn


class Backbone(nn.Module):
    def output_channels(self):
        raise NotImplementedError()

    def output_strides(self):
        raise NotImplementedError()
