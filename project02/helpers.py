import torch
import math
import matplotlib.pyplot as plt

def generator(n_points):
    """ Generate a data set containing n_points sampled uniformly in [0,1]^2.
        Labels are 0 if outside circle of radius 1/sqrt(2*pi) centered at (0.5,0.5),
        Labels are 1 otherwise.

        :param n_points: number of data samples.

        :return data: [(x,y)]0<=n<=n_points-1
        :return labels: vector containing class labels.
    """

    data = torch.empty(n_points, 2).uniform_(0,1)
    dist = data.sub(0.5).pow(2).sum(dim = 1).sub(1 / math.sqrt(2*math.pi))

    labels = dist.sign().add(1).div(2).long() # values in [0,1]

    return data, labels

def plot_cross_validation(accuracy, Adam, path = "/plots/"):
    """
        plotting the cross-validation results, showing the evolution of the accuracy
        of the model(s) with the best hyper-parameters value found with cross-validaiton.

        :param accuracy: dictionnary containing the accuracy and best hyper-parameters
                         values (parameter best_params of class Cross_Validation).
        :param path: path of where to store the plots.
    """

    acc = []
    legend = []

    for name in enumerate(accuracy):
        legend_ = str(name) + " : "
        for name_param in accuracy[name]:
            if name_param == "accuracy":
                acc.append(accuracy[name][name_param])
            else:
                legend_ += str(name_param) + " = " + str(accuracy[name][name_param]) + ", "
        legend.append(legend_)

    plt.figure(figize = (5,5))
    plt.hist(acc)
    if Adam :
        plt.title("Cross Validation Results with Adam Optimizer")
    else :
        plt.title("Cross Validation Results with Stochastic Gradient Descent")
    plt.legend(legend)
    plt.savefig(path + "cross_val_results.png",dpi = "figure", format = "png")
