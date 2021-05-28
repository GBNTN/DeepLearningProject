import torch
from train import *
from modules import *
from loss import LossMSE, CrossEntropy, LossMAE
from helpers import generator

class Cross_Validation:
    def __init__(self, models, names, cross_params = None):
        """ Constructor of the Cross validation module. """

        self.models = models
        self.names = names
        self.best_params = {name : {} for name in self.names}

        if cross_params is None:
            self.cross_params = {
                "lr" : torch.linspace(1e-4, 1e-1, 10),
                "eps" : torch.linspace(1e-8, 1e-6, 4),
                "b1" : torch.linspace(0.8, 0.9, 4),
                "b2" : torch.linspace(0.9, 0.999, 4),
            }
        else :
            self.cross_params = cross_params

    def cross_validation(self, epochs = 100, mini_batch_size = 4, criterion = "MSE", Adam = True):
        """ Cross-validation of the models parameters to find the best set of hyper-parameters.
        """
        train_input, train_labels = generator(1000)
        test_input, test_labels = generator(1000)

        accuracy = torch.zeros((len(self.names),1))

        for lr, eps, b1, b2 in self.iterator(Adam):
            if Adam:
                print("Validation with values : lr = {:.4f}, eps = {:.8f}, b1 = {:.3f}, b2 = {:.3f}".format(lr, eps, b1, b2))
            else:
                print("Validation with values : lr = {:.4f}".format(lr))

            optimizer = Optimizer(models = self.models, names = self.names, epochs = epochs, mini_batch_size = mini_batch_size,
                                  criterion = criterion, learning_rate = lr, Adam = Adam, epsilon = eps,
                                  beta_1 = b1, beta_2 = b2)

            optimizer.train(train_input, train_labels, verbose = False)

            acc = optimizer.compute_accuracy(test_input, test_labels)

            for index, name in enumerate(self.names):
                # Determining the best combination of parameters:
                if acc[index] > accuracy.max(dim = 1)[0][index]:
                    self.best_params[name] = {"lr" : lr,
                                              "eps" : eps,
                                              "b1" : b1,
                                              "b2" : b2,
                                              "accuracy" : acc[index].item()}

                # Printing accuracy scores
                print("Accuracy of the {} = {:.2f}".format(name, acc[index].item()))

            accuracy = torch.cat((accuracy, acc), 1)

        """
        best_accs = accuracy.max(dim = 1)[0]
        for index, name in enumerate(self.names):
            print("Best accuracy found for the {} = {:.2f}".format(name, best_accs[index].float()))
        """

    def set_params(self):
        """ Setting the parameters of the models to the best values found with
            cross validation.
        """

        raise NotImplementedError

    def iterator(self, Adam):
        """ Definition of the iterator function over the different hyper-paraameters
            values to test.
        """

        if Adam:
            iterator = [(lr, eps, b1, b2) for lr in self.cross_params["lr"]
                                          for eps in self.cross_params["eps"]
                                          for b1 in self.cross_params["b1"]
                                          for b2 in self.cross_params["b2"]]
        else:
            iterator = [(lr, 1e-8, 0.9, 0.999) for lr in self.cross_params["lr"]]

        return iterator
