import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):

        # Super class constructor
        super(LinearRegression, self).__init__()
        # define Linear model
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Forward pass
        out = self.linear(x)
        return out


def torch_regression(x_train, x_val, y_train, y_val):
    input_dim = 1
    output_dim = 1

    model = LinearRegression(input_dim, output_dim)

    criterion = nn.MSELoss()  # Mean Squared Loss
    l_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=l_rate)  # Stochastic Gradient Descent

    epochs = 2000

    # Train
    for epoch in range(epochs):
        epoch += 1
        # increase the number of epochs by 1 every time
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

        # clear grads as discussed in prev post
        optimiser.zero_grad()
        # forward to get predicted values
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # back props
        optimiser.step()  # update the parameters
        print('epoch {}, loss {}'.format(epoch, loss.data[0]))

        predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()

        plt.plot(x_train, y_train, 'go', label='from data', alpha=.5)
        plt.plot(x_train, predicted, label='prediction', alpha=0.5)
        plt.legend()
        plt.show()
        print(model.state_dict())


if __name__ == '__main__':
    print()
