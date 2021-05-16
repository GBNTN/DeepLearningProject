import torch
import math

def generator(n_points):
    """ Generate a data set containing n_points sampled uniformly in [0,1]^2.
        Labels are 0 if outside circle of radius 1/sqrt(2*pi) centered at (0.5,0.5),
        Labels are 1 otherwise.
    """

    data = torch.empty(n_points, 2).uniform_(0,1)
    labels = data.sub(0.5).pow(2).sum(dim = 1).sub(1 / math.sqrt(2*math.pi)).sign().add(1).div(2)

    return data, labels
