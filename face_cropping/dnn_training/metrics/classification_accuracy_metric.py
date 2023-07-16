class ClassificationAccuracyMetric:
    def __init__(self):
        self._good = 0
        self._total = 0

    def clear(self):
        self._good = 0
        self._total = 0

    def add(self, predicted_class_scores, target_classes):
        predicted_classes = predicted_class_scores.argmax(dim=1)

        self._good += (predicted_classes == target_classes).sum().item()
        self._total += target_classes.size()[0]

    def get_accuracy(self):
        if self._total == 0:
            return 0
        return self._good / self._total


class TopNClassificationAccuracyMetric:
    def __init__(self, n):
        self._n = n
        self._good = 0
        self._total = 0

    def clear(self):
        self._good = 0
        self._total = 0

    def add(self, predicted_class_scores, target_classes):
        top_n_predicted_classes = predicted_class_scores.argsort(dim=1, descending=True)[:, :self._n]

        for i in range(self._n):
            self._good += (top_n_predicted_classes[:, i] == target_classes).sum().item()
        self._total += target_classes.size()[0]

    def get_accuracy(self):
        if self._total == 0:
            return 0
        return self._good / self._total
