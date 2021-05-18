import torch
from loss import LossMSE

"""
Simple structure : allows to implement several modules and losses that inherit
from it.
"""
class Module(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def init_params(self, Xavier, Xavier_gain):
        raise NotImplementedError

    def stochastic_gradient_descent(self, learning_rate):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

def Linear(Module):
    def __init__(self, input_size, output_size, Xavier = True, Xavier_gain = 1.0):
        """ Constructor of the Linear layer class.

            :param input_size: input dimension.
            :param output_size: output dimension.
            :param Xavier: Boolean, will intialize the weight and bias parameter
                           according to a normal distribution with the Xavier method
                           if set to True (standard deviation changed). Otherwise,
                           it will initialize the latter with a std of 1.0.
            :param Xavier_gain: gain of the std, with the Xavier method.
        """

        Module.__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.input = None

        self.w = zeros((input_size, output_size))
        self.b = zeros((1, output_size))

        self.grad_w = None
        self.grad_b = None

        self.mean_w = zeros((input_size, output_size))
        self.var_w = zeros((input_size, output_size))

        self.mean_b = zeros((1, output_size))
        self.var_b = zeros((1, output_size))

        self.init_params(Xavier, Xavier_gain)

    def forward(self, *input):
        """ Forward pass trough the layer of the Linear class.

            :param input: the input(s) of the layer.

            :return: the output of the layer (prediction).
        """
        input_ = input[0].clone()

        self.input = input_

        return input_.mm(self.w) + self.b

    def backward(self, *gradwrtoutput):
        """ backpropagation of the output gradient of the layer of the Linear class.

            :param gradwrtoutput: gradient with respect to the output of layer.

            :return: backpropagation of the output gradient.
        """
        input_ = gradwrtoutput[0].clone()
        grad = self.input.t().mm(input_)

        self.grad_w += grad
        self.grad_b += input_.sum(0)

        return grad

    def param(self):
        """ Return the parameters of the layer (weight values and gradient, bias
            values and gradient).
        """
        return [(self.w, self.grad_w, self.mean_w, self.var_w),
                (self.b, self.grad_b, self.mean_b, self.var_b)]

    def init_params(Xavier, Xavier_gain):
        """ initliazation of the weights and bias parameters of the layer.

            :param Xavier: A boolean, which enables the xavier initliazation of
                           weights and bias parameters, to resolve vanishing gradient issue.
            :param Xavier_gain: Gain of the Xavier standard deviation.
        """
        if Xavier:
            std = Xavier_gain * math.sqrt(2.0 / (self.input_size + self.output_size))
        else:
            std = 1.0

        self.w = torch.empty((self.input_size,self.output_size)).normal_(0, std)
        self.b = torch.empty((1,self.output_size)).normal_(0, std)

    def gradient_descent(self, learning_rate = 0.001):
        """ Gradient descent of layer of Linear class.

            :param learning_rate: learning rate.
        """

        self.w.sub_(learning_rate * self.grad_w)
        self.b.sub_(learning_rate * self.grad_b)

    def zero_grad(self):
        """ Put the gradient of each layer to zero."""

        self.w.new_zeros(self.w.shape)
        self.b.new_zeros(self.b.shape)


def Sequential(Module):
    def __init__(self, *layers, Xavier = True, Xavier_gain = 1.0):
        """ Constructor of the Sequential model, which can stack multiple layers.

            :param layers: the different layer(s) of the neural network model.
            :param Xavier: Boolean, will intialize the weight and bias parameter
                           according to a normal distribution with the Xavier method
                           if set to True (standard deviation changed). Otherwise,
                           it will initialize the latter with a std of 1.0.
            :param Xavier_gain: gain of the std, with the Xavier method.
        """
        Module.__init__()
        self.layers = list(layers)

        for layer in self.layers :
            layer.init_params(Xavier, Xavier_gain)

    def forward(self, *input):
        """ Forward pass trough the different layer(s).

            :param input: the input(s) of the layer(s).

            :return: the output of the last layer (prediction).
        """
        output = input[0].clone()

        for layer in layers:
            output = layer.forward(output)

        return output

    def backward(self, *gradwrtoutput):
        """ backpropagation of the output gradient through the layers.

            :param gradwrtoutput: gradient with respect to the output of the
                                  different layers.

            :return: backpropagation of the output gradient.
        """
        bw = gradwrtoutput[0].clone()

        for i in range(len(layers)-1, 0, -1):
            bw = layers[i].backward(bw)

        return bw

    def param(self): # TODO
        """ Return the parameters of each layer in an array of the size of the
            number of layer.
        """

        param = []

        for layer in layers:
            param.append([(self.w, self.grad_w, self.mean_w, self.var_w),
                          (self.b, self.grad_b, self.mean_b, self.var_b)])

        return param

    def gradient_descent(self, learning_rate = 0.001):
        """ Gradient descent of each layer.

            :param learning_rate: learning rate.
        """

        for layer in self.layers:
            layer.gradient_descent(learning_rate)

    def zero_grad(self):
        """ Put the gradient of each layer to zero."""

        for layer in self.layers:
            layer.zero_grad()
