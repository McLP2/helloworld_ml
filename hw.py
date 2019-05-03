import time

import torch
import torch.optim
import torch.nn as nn


def text_to_vector(string):
    char_values = []
    for char in list(string):
        char_values.append(ord(char)+.5)
    return torch.tensor(char_values, dtype=torch.float)


def vector_to_text(tensor):
    string = ""
    for value in tensor:
        string += chr(value)
    return string


def train():
    goal = "Hello World!"
    learning_rate = 0.001
    input_size = 5
    output_size = len(goal)
    goal_vector = text_to_vector(goal)

    class MLP(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 8),
                nn.ReLU(),
                nn.Linear(8, output_size)
            )

        def forward(self, x):
            return self.layers(x)

    e = 0

    model = MLP(input_size)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)

    while True:
        output = model.forward(torch.rand((input_size,)))
        loss = loss_function(output, goal_vector)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        e += 1
        if e % 5 == 0:
            result = vector_to_text(output)
            print(result)
            time.sleep(0.2)


if __name__ == '__main__':
    train()
