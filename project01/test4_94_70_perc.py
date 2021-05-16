import os; os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from dlc_practical_prologue import *


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1*14*14, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.log_softmax(self.layer4(x), dim=1)
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

    # Put all the numbers in just one column
    train_input = torch.cat([train_input[:, :1*14*14], train_input[:, 1*14*14:]],dim=0)
    train_class = torch.cat([train_classes[:, 0], train_classes[:, 1]],dim=0)

    test_input = torch.cat([test_input[:, :1 * 14 * 14], test_input[:, 1 * 14 * 14:]], dim=0)
    test_class = torch.cat([test_classes[:, 0], test_classes[:, 1]], dim=0)

    # Create the neural network
    net = Net()


    # Training
    learning_rate = 0.001
    # Use Adam optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    EPOCHS = 300

    for epoch in range(EPOCHS):
        output = net(train_input)
        loss = nn.functional.nll_loss(output, train_class)
        #print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print(l)


    # Testing class prediction
    
    with torch.no_grad():
        output = net(test_input)
        class_predicted = torch.argmax(output, dim=1)

        total = 2000
        correct = (test_class == class_predicted).sum()

        print("Accuracy of class prediction %.2f%%" % (correct / total * 100))
    
        output = net(test_input)
        class_predicted = torch.argmax(output, dim=1)
        classes_predicted = torch.transpose(class_predicted.view(2,1000), 0, 1)

        target_predicted = (classes_predicted[:, 0] <= classes_predicted[:, 1]).view(-1,1)

        #(test_target == target_predicted)

        #print(test_target == 1)
        #print(target_predicted.sum(dim=1).view(-1,1))
        total = 1000
        correct = (test_target == target_predicted).sum()
        print("Accuracy of target prediction %.2f%%" % (correct / total * 100))