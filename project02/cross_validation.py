import torch
from train import *
from models import *
from loss import LossMSE
from helpers import generator

class Cross_Validation:
    def __init__(self, models, names, cross_params = None):
        """ Constructor of the Cross validation module. """

        self.models = list(models)
        self.names = names
        self.best_params = {name : {} for name in self.names}

        if test_params is None:
            self.cross_params = {
                "lr" : np.linspace(1e-4, 1e-1, 5e-4),
                "eps" : np.linspace(1e-8, 1e-6, 5e-8),
                "b1" : np.linspace(0.8, 0.9, 0.1),
                "b2" : np.linspace(0.9, 0.999, 0.01, endpoint = True),
            }
        else :
            self.cross_params = cross_params

    def cross_validation(self, epochs = 100, mini_batch_size = 4, criterion = "MSE", Adam = True):
        """ Cross-validation of the models parameters to find the best set of
            params.
        """
        train_input, train_labels = generator(1000)
        test_input, test_labels = generator(1000)

        accuracy = zeros((1,len(self.names)))

        for lr, eps, b1, b2 in self.iterator(Adam):
            if Adam:
                print("Validation with values : lr = {}, eps = {}, b1 = {}, b2 = {}".format(learning_rate, epsilon, beta_1, beta_2))
            else:
                print("Validation with values : lr = {}".format(learning_rate))

            optimizer = Optimizer(self.models, self.names, epochs, mini_batch_size, criterion,
                                  learning_rate = lr, Adam, epsilon = eps, beta_1 = b1, beta_2 = b2)

            optimizer.train(train_input, train_labels, verbose = False)

            acc = optimizer.compute_accuracy(test_input, test_labels)

            for index, name in enumerate(self.names):
                # Determining the best combination of parameters:
                if acc[index] > accuracy.min(dim = 1)[index
                    self.best_params[name] = {"lr" : lr,
                                              "eps" : eps,
                                              "b1" : b1,
                                              "b2" : b2,
                                              "accuracy" = acc}

                # Printing accuracy scores
                print("Accuracy of the {} = {}".format(name, acc))

            accuracy.cat((accuracy, acc), 1)


    def set_params(self):
        """ Setting the parameters of the models to the best values found with
            cross validation.
        """

        raise NotImplementedError

    def iterator(self, Adam):
        """ Definition of the iterator function over the different hyper-paraameters
            values to test.
        """

        if Adam = True:
            iterator = [(lr, eps, b1, b2) for lr in self.cross_params["lr"]
                                          for eps in self.cross_params["eps"]
                                          for b1 in self.cross_params["b1"]
                                          for b2 in self.cross_params["b2"]]
        else:
            iterator = [(lr, 1e-8, 0.9, 0.999) for lr in self.cross_params["lr"]]

        return iterator
