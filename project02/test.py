import torch
import math

from loss import *
from helpers import generator, plot_cross_validation
from modules import *
from activations import ReLU, Tanh, Sigmoid
from cross_validation import Cross_Validation

torch.set_grad_enabled(False)

# Parameters :
N = 1000
INPUT_SIZE = 2
OUTPUT_SIZE = 2
NUM_HIDDEN_LAYERS = 3
NUM_HIDDEN_UNITS = 25
NUM_EPOCH = 100
BATCH_SIZE = 10
CRITERION = "MSE"

# Adam Optimizer parameters : (best params found with cross-validation)
ADAM = True
LEARNING_RATE = 0.001
B1 = 0.9
B2 = 0.09
EPSILPON = 0.01

# Cross validation boolean parameter :
CROSS_VALIDATION = True
PLOT = True # Valid only if param CROSS_VALIDATION is True.

# Generate Data sampled from an uniform distribution in the interval [0,1]
train_input, train_labels = generator(N)
test_input, test_labels = generator(N)


Models_names = ["RelU_network", "Tanh_network", "Sigmoid_network"]
Models = [Sequential(Linear(INPUT_SIZE,NUM_HIDDEN_UNITS), ReLU(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), ReLU(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), ReLU(),
                     Linear(NUM_HIDDEN_UNITS,OUTPUT_SIZE), Xavier = True),
          Sequential(Linear(INPUT_SIZE,NUM_HIDDEN_UNITS), Tanh(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), Tanh(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), Tanh(),
                     Linear(NUM_HIDDEN_UNITS,OUTPUT_SIZE), Xavier = True),
          Sequential(Linear(INPUT_SIZE,NUM_HIDDEN_UNITS), Sigmoid(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), Sigmoid(),
                     Linear(NUM_HIDDEN_UNITS,NUM_HIDDEN_UNITS), Sigmoid(),
                     Linear(NUM_HIDDEN_UNITS,OUTPUT_SIZE), Xavier = True)
         ]

if CROSS_VALIDATION:
    print("Cross-Validation of the hyperparameters")
    cross_params = {"lr" : torch.linspace(1e-4, 1e-1, 10)}

    if ADAM:
        cross_params["eps"] = torch.linspace(1e-8, 1e-6, 5)
        cross_params["b1"] = torch.linspace(0.8, 0.9, 10)
        cross_params["b2"] = torch.linspace(0.9, 0.999, 10)

    CV = Cross_Validation(models = Models, names = Models_names, cross_params = cross_params)
    CV.cross_validation(epochs = NUM_EPOCH, mini_batch_size = BATCH_SIZE,
                        criterion = CRITERION, Adam = ADAM)

    print("Results of Cross-Validation")
    for model, name in zip(Models, Models_names):
        print("The best parameters of the {} with an accuracy of {} are :".format(name, CV.best_params[name]["accuracy"]))
        for param_name in CV.best_params[name]:
            if not param_name == "accuracy":
                print("{} = {} ".format(param_name, CV.best_params[name][param_name]))

    if PLOT:
        plot_cross_validation(ADAM, CV.best_params)
else :
    # Construct the optimizer and generate Data sampled from an uniform distribution in the interval [0,1]
    optimizer = Optimizer(Models, Models_names, epochs = NUM_EPOCH,  mini_batch_size = BATCH_SIZE,
                          criterion = CRITERION, learning_rate = LEARNING_RATE, Adam = ADAM,
                          epislon = EPSILON, beta_1 = B1, beta_2 = B2)

    # Training of the models:
    optimizer.train()

    # Computing the accuracy :
    accuracy_train = optimizer.compute_accuracy(train_input, train_labels)
    accuracy_test = optimizer.compute_accuracy(test_input, test_labels)

    for index, name in enumerate(Models_names):
        print('Train accuracy of {} = {:.2f}'.format(name, accuracy_train[index].item()*100))
        print('Test accuracy of {} = {:.2f}'.format(name, accuracy_test[index].item()*100))
