from torch import zeros

class LossMSE(Module):
    def __init__(self):
        Module.__init__()
        self.error = None
        self.pred = None
        self.labels = None

    def forward(self, labels, pred):
        self.pred = pred
        self.labels = labels

        # Error between predicted labels and targets
        self.error = self.pred - self.labels

        return self.error.pow(2).sum()


    def backward(self):
        return 2 * self.error / self.error.size()
