import torch
from modules import Module

class LossMSE(Module):
    def __init__(self):
        """ Constructor of the Mean Squared Error (MSE) loss function. """
        Module.__init__(self)
        self.error = None
        self.preds = None
        self.labels = None

    def forward(self, preds, labels):
        """ Compute the MSE Loss function of the predictions with
            respect to the true labels.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: MSE.
        """
        self.preds = preds.clone()
        self.labels = torch.zeros((labels.shape[0], 2)).scatter_(1, labels.view(-1, 1), 1)

        # Error between predicted labels and targets
        self.error = self.preds - self.labels

        return self.error.pow(2).mean()

    def backward(self):
        """ Compute the loss function (MSE) gradient. """
        return 2 * self.error / self.preds.size(0)

class CrossEntropy(Module):
    def __init__(self):
        """ Constructor of the cross entropy loss function, which is defined by the
            logarithm of the MSE.
        """

        Module.__init__(self)
        self.preds = None
        self.labels = None

    def softmax(self, input):
        """ definition of the softmax function.

            :param input: input of the function.

            :return: the softmax activation with respect to the input.
            formula = e^(x) / sum(e^x)
        """
        max = input.max(dim = 0)[0]

        e_x = torch.exp(input.sub_(max))

        return e_x.div(e_x.sum())

    def forward(self, preds, labels):
        """ forward pass of the cross entropy, that calculation of the prediction
            error.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: Cross Entropy error.
        """

        self.preds = preds.clone()
        self.labels = torch.zeros((labels.shape[0], 2)).scatter_(1, labels.view(-1, 1), 1)

        loss = - self.labels * torch.log(self.softmax(self.preds))
        return loss.sum() / self.preds.size(0)

    def backward(self):
        """ Backward pass of the cross-entropy function, that is the gradient. """

        grad = self.softmax(self.preds)

        return grad - self.labels


class LossMAE(Module):
    def __init__(self):
        """ Constructor of the Mean Absolute Error (MSE) loss function. """
        Module.__init__(self)
        self.error = None
        self.preds = None
        self.labels = None

    def forward(self, preds, labels):
        """ Compute the MAE Loss function of the predictions with
            respect to the true labels.

            :param labels: True class labels.
            :param pred: Predicted class labels by a model.

            :return: MAE.
        """
        self.preds = preds

        # labels to one-hot-vectors :
        self.labels = torch.zeros((labels.size(0), 2)).scatter_(1, labels.view(-1, 1), 1)

        # Error between predicted labels and true labels :
        self.error = self.preds - self.labels
        self.error[self.error < 0] = -self.error[self.error < 0]

        return self.error.mean()


    def backward(self):
        """ Compute the loss function (MAE) gradient. """
        return self.error.sign() / self.error.size()
