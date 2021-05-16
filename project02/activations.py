import torch
from modules import Module

class Relu(Module):
    def __init__():
        Module.__init__(self)
        self.s = None

    def forward(self, *input):
        s = input[0].clone()
        self.s = s

        s[s<0] = 0

        return s

    def backward(self, *gradwrtoutput):
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        bw[bw < 0] = 0
        bw[bw > 0] = 1

        return bw @ input


class Tanh(Module):
    def __init__():
        Module.__init__(self)
        self.s = None

    def tanh(x):
        return (math.e**x - (1/math.e)**(-x)) / (math.e**x + (1/math.e)**(-x))

    def forward(self, *input):
        """ return the hyperbolic tangent activation of input
            formula := (math.e**s - (1/math.e)**(-s)) / (math.e**s + (1/math.e)**(-s))
        """
        s = input[0].clone()
        self.s = s

        return tanh(s)

    def backward(self, *gradwrtoutput):
        """ return the hyperbolic tangent gradient with respect to the input
            formula := (1 - tanh(x_i)^2) * grad(x_i+1)
        """
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        return (1 - tanh(bw)**2) @ input


class Sigmoid(Module):
    def __init__():
        Module.__init__(self)
        self.s = None

    def sigmoid_(x):
        return 1 / (1 + math.e ** (-x))

    def forward(self, *input):
        s = input[0].clone()
        self.s = s

        return sigmoid_(s)

    def backward(self, *gradwrtoutput):
        input = gradwrtoutput[0].clone()
        bw = self.s.clone()

        grad = sigmoid_(bw)*(1 - sigmoid_(bw))

        return grad @ input
