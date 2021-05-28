# DeepLearningProject - EE-559
MiniProjects 1 and 2

## Project 01

The objective of this project is to test different architectures to compare two digits visible in a two-channel image.

### Data

The training and test sets consist of 1000 pairs of 14x14 greyscale images each depicting a digit. Every pair is labelled by a 1 if the first digit is lower or equal to the second, 0 otherwise.

### Methods 

We tried to solve the problem using two strategies. 

In the first strategy, we converted the two 14x14 greyscale pictures into a vector of size 392, that is 14 times 14 times 2, and we created a network made of linear layers that outputs one value which is 1 if the first digit is smaller or equal to the second one, 0 otherwise. 

In the second strategy, we created a neural network that takes just one picture at a time, converts it into a number, and only then the numbers are compared to conclude whether the first pictures shows a smaller or equal number compared to the one in the second picture.

### Libraries and Tools

We used Python 3.9 version.

* [Pytorch](https://pytorch.org)
* dlc_practical_prologue.py


## Project 02

The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

### Data
The training and test sets consist of 1000 data points sampled from the uniform distribution in $[0,1]^2$, labeled each with 1 if they are inside a disk of radius $1/ \sqrt{2 \pi}$ centered at (0.5, 0.5), and 0 otherwise. 

### Methods 

We implemented a variety of modules such that the user can play with different architectures. Thus, the user can choose from the following options :

- Activation functions:
  - Rectified Linear Unit (ReLU),
  - Hyperbolic tangent (Tanh),
  - Sigmoid.
- Loss functions:
  - Mean Squared Error (MSE),
  - Meaan Absolute Error (MAE),
  - Cross Entropy (CE).
- Choice of Optimization :
  - Stochastic gradient descent (SGD),
  - Adam.

### How to run the code:

There are 2 ways of running the code in project 2 : the user can either run directly the <code>test.py</code> file or use the google colab notebook <code>DLP_02.ipynb</code>. The notebook allows one to run the code interactively. The parameters to play with are the following:

> **N**: the size of the train and test sets (set to 1000) <br/>
> **INPUT_SIZE**: input size of the model (set to 2) <br/>
> **OUTPUT_SIZE**: output size of the model (set to 2) <br/>
> **NUM_HIDDEN_LAYERS**: number of hidden layers of the model (set to 3) <br/>
> **NUM_HIDDEN_UNITS**: number of hidden units per layer (set to 25) <br/>
> **NUM_EPOCH**: number of epochs on which to train the model(s) (set to 100) <br/>
> **BATCH_SIZE**: batch size (set to 10) <br/>
> **CRITERION**: loss function ("MSE", "MAE", or "CE") <br/>

Weight initialization method:
> XAVIER_GAIN : the gain of the standard deviation used in Xavier method (set to 6.0) <br/>

Parameters for Adam Optimizer parameters:
> **ADAM**: will use Adam optimization if set to True, and SGD otherwise. <br/>
> **LEARNING_RATE**: learning rate (set to 0.0001) <br/>
> **B1**: (set to 0.8) <br/>
> **B2**: (set to 0.899) <br/>
> **EPSILPON**: prevent from division by zero (set to 1e-8) <br/>

Cross validation boolean parameter :
> **CROSS_VALIDATION**: the program will perform cross-validation of the hyperparameters (learning rate, B1, B2, epsilon) if True, and train and test the tree model (with ReLU, Tanh, and Sigmoid activations) with best parameters values otherwise. <br/>
> **PLOT**: the program will plot the result of cross-validation if True (valid only if parameter <code>CROSS_VALIDATION</code> is True). <br/>

### Libraries and Tools

We used Python 3.7 version.

* [Pytorch](https://pytorch.org)
* [Math](https://docs.python.org/3/library/math.html)

