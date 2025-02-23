import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Block, self).__init__()

        self.fc_1a = nn.Linear(input_dim, output_dim)
        self.fc_1b = nn.Linear(output_dim, output_dim)

        self.fc_2a = nn.Linear(input_dim, output_dim)
        self.fc_2b = nn.Linear(output_dim, output_dim)
        self.concat = nn.Linear(output_dim + output_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.cat((self.fc_1b(self.fc_1a(x)), self.fc_2b(self.fc_2a(x))), 2)
        x = self.relu(self.concat(x))
        return x
