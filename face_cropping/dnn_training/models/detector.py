import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, backbone, neck, head, output_decoded_predictions=False):
        super().__init__()

        self._backbone = backbone
        self._neck = neck
        self._head = head

        self._output_decoded_predictions = output_decoded_predictions

    def forward(self, x):
        features_maps = self._backbone(x)
        predictions, priors = self._head(self._neck(features_maps))

        if self._output_decoded_predictions:
            return self.decode_predictions(predictions, priors)
        else:
            return predictions, priors

    def decode_predictions(self, predictions, priors):
        return self._head.decode_predictions(predictions, priors)
