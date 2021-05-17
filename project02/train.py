import torch
import math
import cross_validation
from helpers import generator

class Train:
    def __init__(self, *models, epochs = 100,  batch_size = 16, learning_rate = 0.001,
                 epislon = 0.01, beta_1 = 0.9, beta_2 = 0.09):
        """ Constructor of the Train class, enables to train multiple networks.

            :param models:
            :param epochs:
            :param batch_size:
            :param learning_rate:
            :param epsilon:
            :param beta_1:
            :param beta_2:

            :return: the models and their accuracy in a dictionnary.
        """

        self.models = list(models)

        # Parameters for duration:
        self.epochs = epochs
        self.batch_size = batch_size

        # Parameters for the optimization (with Adam):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step = 0

        # Generate train and test data:
        self.train_input, self.train_labels = generator(1000)
        self.test_input, self.test_labels = generator(1000)

        # Training the model(s):
        train()

        # Computing the accuracy of the model(s):
        self.accuracy = []
        compute_accuracy(self.input, self.labels)

        return self.models, self.accuracy

    # TO BE FINISHED
    def create_mini_batches(self):
        """ create mini batches to be iterated over while training the model(s). """
        mini_batches = []
        num_batches = self.train_input.size(0) // self.batch_size

        for index in range(0, train_input.size(0), ):
            batch_indices = tensor()
            mini_batch = self.train_input.narrow()

        return minibatches

    # TO BE FINISHED
    def train(self):
        """ Training of the model(s).

        """
        for model in models:
            for epoch in range(self.epochs):

                for batch in self.create_mini_batches():

        raise NotImplementedError

    def update(self):
        """ Update of the weight and bias parameters of the model(s). """

        for model in self.models:
            model.stochastic_gradient_descent(learning_rate = learning_rate)

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

    def compute_accuracy(self):
        """ Compute the model(s) prediction accuracy.

            :param input:
            :param output:
        """
        for model in self.models:
            predicted_labels = self.model.forward(self.test_input)
            grad = self.model.zero_grad()
            self.accuracy.append((predicted_labels == self.test_labels).mean())
