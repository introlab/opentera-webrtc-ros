class LossMetric:
    def __init__(self):
        self._loss = 0
        self._count = 0

    def clear(self):
        self._loss = 0
        self._count = 0

    def add(self, loss):
        self._loss += loss
        self._count += 1

    def get_loss(self):
        if self._count == 0:
            return 0
        return self._loss / self._count
