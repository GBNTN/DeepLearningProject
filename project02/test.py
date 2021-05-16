import torch
import math

from loss import *
from helpers import generator
from modules import *

torch.set_grad_enabled(False)

# Parameters :
N = 1000
INPUT_SIZE = 8
OUTPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 3
NUM_HIDDEN_UNITS = 25

learning_rate = 0.001

# Generate Data sampled from an uniform distribution in the interval [0,1]
X_train, Y_train = generator(N)
X_test, Y_test = generator(N)

Model = Sequential(Linear(2,25), Relu(),
                   Linear(25,25), Relu(),
                   Linear(25,25), Relu(),
                   Linear(25,1), Sigmoid())
