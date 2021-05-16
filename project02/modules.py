import torch
from loss import LossMSE

class Module(object):
    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

def Linear(Module):
    def __init__(self, input_size, output_size, Xavier = True, Xavier_gain = 1.0):
        Module.__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.input = None

        self.w = zeros((input_size, output_size))
        self.b = zeros((1, output_size))

        self.grad_w = None
        self.grad_b = None

        self.init_params(Xavier, Xavier_gain)

    def forward(self, *input):
        input_ = input[0].clone()

        self.input = input_

        return input_.mm(self.w) + self.b

    def backward(self, *gradwrtoutput):
        input_ = gradwrtoutput[0].clone()
        grad = self.input.t().mm(input_)

        self.grad_w += grad
        self.grad_b += input_.sum(0)

        return grad

    def param(self):
        return [self.w, self.grad_w, self.b, self.grad_b]

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


def Sequential(Module):
    def __init__(self, *layers, Xavier = True, Xavier_gain = 1.0):
        Module.__init__()
        self.layers = list(layers)

        for layer in self.layers :
            layer.init_params(Xavier, Xavier_gain)

    def forward(self, *input):
        input_ = input[0].clone()

        for layer in layers:
            input_ = layer.forward(input_)

        return input_

    def backward(self, *gradwrtoutput):
        grad = gradwrtoutput[0].clone()

        for i in range(len(layers)-1, 0, -1):
            grad = layers[i].backward(grad)

        return grad

    def param(self): # TODO
        return []
