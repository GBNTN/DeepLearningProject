import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from dlc_practical_prologue import *


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2*14*14, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x


if __name__ == '__main__':
    # Data initialization
    N = 1000
    train_input, train_target, train_classes, _, _, _, = generate_pair_sets(N)
    _, _, _, test_input, test_target, test_classes = generate_pair_sets(N)

    train_input = train_input.view(-1, 2*14*14)
    test_input = test_input.view(-1, 2*14*14)

    train_target = train_target.view(-1, 1)
    test_target = test_target.view(-1, 1)

    # I convert the type to torch.float32
    train_input, train_target, train_classes, test_input, test_target, test_classes = \
        train_input.type(torch.float32), train_target.type(torch.float32), train_classes.type(torch.long), \
        test_input.type(torch.float32), test_target.type(torch.float32), test_classes.type(torch.long)


    # Create the neural network
    net = Net()


    # Training
    learning_rate = 0.01
    # Use binary cross entropy loss
    loss = nn.MSELoss()
    #loss = nn.BCELoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    EPOCHS = 50

    for epoch in range(EPOCHS):
        target_predicted = net(train_input)
        l = loss(train_target, target_predicted)  #loss = nn.MSELoss()
        #l = loss(target_predicted, train_target)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print(l)


    # Testing
    total = 1000
    correct = 0
    with torch.no_grad():
        correct = ( test_target == net(test_input).round() ).sum()

    print("Accuracy %.2f%%" % (correct / total * 100))