import torch
import math
from modules import Module

class ReLU(Module):
    def __init__(self):
        """ Constructor the Rectified Linear Unit (ReLU) activation function
            module.
        """
        Module.__init__(self)
        self.s = None

    def forward(self, *input):
        """ Implementation of the forward pass through each input with the RelU
            activation function.
        """
        s = input[0].clone()
        self.s = s

        s[s<0] = 0

        return s

    def backward(self, *gradwrtoutput):
        """ Determine the RelU gradient with respect to the output of the episode
            (backpropagation of the gradient).

            :param gradwrtoutput: gradient of the layer with respect to the output.

            :return: backpropagation of the RelU gradient.
        """
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        bw[bw < 0] = 0
        bw[bw > 0] = 1

        return bw.mul(input)


class Tanh(Module):
    def __init__(self):
        """ Constructor of the hyperbolic tangent activation function module. """
        Module.__init__(self)
        self.s = None

    def tanh(self, x):
        """ Definition of the tanh function. """
        return (math.e**x - (1/math.e)**(-x)) / (math.e**x + (1/math.e)**(-x))

    def forward(self, *input):
        """ Implementation of the forward pass through each input with the
            hyperbolic tangent activation function.

            :param input: input of the layer.

            :return: the activation of the input with the hyperbolic tagent function.
        """
        s = input[0].clone()
        self.s = s

        return self.tanh(s)

    def backward(self, *gradwrtoutput):
        """ Determine of the hyperbolic tangent gradient with respect to the output
             of the episode (backpropagation of the gradient).

            :param gradwrtoutput: gradient of the layers with respect to the output.

            :return: backpropagation of the hyperbolice tangent gradient.

            backward pass := (1 - tanh(x_i)^2) * grad(x_i+1)
        """
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        return (1 - self.tanh(bw)**2).mul(input)


class Sigmoid(Module):
    def __init__(self):
        """ Constructor of the sigmoid activation function module. """
        Module.__init__(self)
        self.s = None

    def sigmoid_(self, x):
        """ Definition of the sigmoid function. """
        return 1 / (1 + math.e ** (-x))

    def forward(self, *input):
        """ Implementation of the forward pass through each input with the
            sigmoid activation function.

            :param input: input of the layer.

            :return: the activation of the input with the sigmoid function.
        """
        s = input[0].clone()
        self.s = s

        return self.sigmoid_(s)

    def backward(self, *gradwrtoutput):
        """ Determine the sigmoid gradient with respect to the output of the
            episode (backpropagation of the gradient).

            :param gradwrtoutput: gradient with respect to the output.

            :return: backpropagation of the sigmoid gradient.

            backward pass := (1 - tanh(x_i)^2) * grad(x_i+1)
        """
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        grad = self.sigmoid_(bw)*(1 - self.sigmoid_(bw))

        return grad.mul(input)
