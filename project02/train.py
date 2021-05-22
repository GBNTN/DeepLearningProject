import torch
import math
import cross_validation
import loss
from helpers import generator


class Optimizer:
    def __init__(self, *models, names, epochs = 100,  mini_batch_size = 4, criterion = "MSE", learning_rate = 0.001, Adam = True
                 epislon = 1e-8, beta_1 = 0.9, beta_2 = 0.999):
        """ Constructor of the Train class, enables to train multiple networks.

            :param models: list of the model(s)
            :param epochs: number of epochs
            :param mini_batch_size: mini_batch size
            :param criterion: if "MSE", use the MSE loss; if "MAE" it will be the MAE loss function.
            :param learning_rate: learning rate for optimization of parameters.
            :param epsilon: small value preventing from zero division.
            :param beta_1: hyperparameter for the calculation ot the mean in the Adam optimizer.
            :param beta_2: hyperparameter for the calculation ot the variance in the Adam optimizer.

            :return: the models and their accuracy in a dictionnary.
        """

        # Models and info :
        self.models = list(models)
        self.names = names

        # Parameters for duration:
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        if criterion == "MSE":
            self.criterion = LossMSE()
        elif criterion == "MAE":
            self.criterion == LossMAE()

        # Parameters for the optimization (with Adam):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step = 0

        """
        # Training the model(s):
        train()

        # Computing the accuracy of the model(s):
        self.accuracy = []
        compute_accuracy(self.input, self.labels)

        return self.models, self.accuracy
        """

    def train(self, train_input, train_labels, verbose = False):
        """ Training of the model(s) with either stochastic gradient descent or
            with adam optimizer if param Adam is True.
        """
        for model, name in zip(self.models, self.names):
            print('Training {}...'.format(name))
            for epoch in range(self.epochs):
                loss = 0.0

                for batch_index in range(0, train_input.size(0), self.mini_batch_size):
                    batch_input = train_input.narrow(dim = 0, batch_index, self.mini_batch_size)
                    batch_labels = train_labels.narrow(dim = 0, batch_index, self.mini_batch_size)

                    model.zero_grad()

                    pred = model.forward(batch_input)
                    loss += self.criterion.forward(pred, batch_labels)

                    gradwrtoutput = self.criterion.backward()
                    self.model.backward(gradwrtoutput)

                    if Adam:
                        self.adam_optimizer()
                    else:
                        self.stochastic_gradient_descent()

                min_loss = loss.min()
                epoch_min_loss = loss.argmin()

                if verbose:
                    print('Epoch = {}, {} Loss = {}, Best Epoch = {}, Best Val = {}'.format(epoch, self.criterion, loss, epoch_min_loss, min_loss))


    def stochastic_gradient_descent(self):
        """ Update of the weight and bias parameters of the model(s). """
        self.t += 1

        for model in self.models:
            model.gradient_descent(learning_rate = self.learning_rate)

    def adam_optimizer(self):
        """ optimization with Adam. """

        self.step += 1

        for model in self.models:
            for (w_b, grad, mean, var) in model.param():
                mean = self.beta_1 * mean + (1-self.beta_2) * grad
                var = self.beta_2 * var + (1-self.beta_2) * grad**2

                mean_hat = mean / (1 - beta_1**(self.step + 1))
                var_hat = var / (1 - beta_2**(self.step + 1))

                w_b.sub_(self.learning_rate * mean_hat / (math.sqrt(var_hat) + epsilon))

    def compute_accuracy(self, test_input, test_labels):
        """ Compute the model(s) prediction accuracy. """
        accuracy = []

        for model in self.models:
            predicted_labels = self.model.forward(test_input)
            grad = self.model.zero_grad()
            accuracy.append((predicted_labels.argmax(dim = 1) == test_labels).mean())

        return accuracy
