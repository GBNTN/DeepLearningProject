from torch import zeros

class LossMSE(Module):
    def __init__(self):
        """ Constructor of the Mean Squared Error (MSE) loss function. """
        Module.__init__()
        self.error = None
        self.pred = None
        self.labels = None

    def forward(self, pred, labels):
        """ Compute the MSE Loss function of the predictions with
            respect to the true labels.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: MSE.
        """
        self.pred = pred
        self.labels = zeros((labels.size(0), 2)).scatter(1, labels.view(-1,1), 1.0)

        # Error between predicted labels and targets
        self.error = self.pred - self.labels

        return self.error.pow(2).mean()

    def backward(self):
        """ Compute the loss function (MSE) gradient. """
        return 2 * self.error / self.error.size()


class LossMAE(Module):
    def __init__(self):
        """ Constructor of the Mean Absolute Error (MSE) loss function. """
        Module.__init__()
        self.error = None
        self.pred = None
        self.labels = None

    def forward(self, pred, labels):
        """ Compute the MAE Loss function of the predictions with
            respect to the true labels.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: MAE.
        """
        self.pred = pred

        # labels to one-hot-vectors :
        self.labels = zeros((labels.size(0), 2)).scatter(1, labels.view(-1,1), 1.0)

        # Error between predicted labels and true labels :
        self.error = self.pred - self.labels
        self.error[self.error < 0] = -self.error[self.error < 0]

        return self.error.mean()


    def backward(self):
        """ Compute the loss function (MAE) gradient. """
        return self.error.sign() / self.error.size()
