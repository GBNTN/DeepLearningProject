from torch import zeros

class LossMSE(Module):
    def __init__(self):
        """ Constructor of the Mean Squared Erro (MSE) loss function. """
        Module.__init__()
        self.error = None
        self.pred = None
        self.labels = None

    def forward(self, pred, labels):
        """ Compute the Mean Squared error Loss function of the predictions with
            respect to the true labels.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: MSE.
        """
        self.pred = pred
        self.labels = labels

        # Error between predicted labels and targets
        self.error = self.pred - self.labels

        return self.error.pow(2).sum()


    def backward(self):
        """ Compute the loss function (MSE) gradient. """
        return 2 * self.error / self.error.size()
