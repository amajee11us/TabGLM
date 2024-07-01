import torch
import torch.nn as nn
import numpy as np

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
        x = torch.cat((self.fc_1b(self.fc_1a(x)),
                       self.fc_2b(self.fc_2a(x))), 2)
        x = self.relu(self.concat(x))
        return x

class GraphNetwork(nn.Module):
    '''
    Code for this block is heavily adapted from https://github.com/amrmalkhatib/IGNNet
    '''
    def __init__(self, input_dim, num_features, adj):
        super(GraphNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)

        self.block1 = Block(64, 64)

        self.block2 = Block(64, 128)

        self.fc4 = nn.Linear(64 + 128, 256)
        self.bn1 = nn.BatchNorm1d(num_features)

        self.block3 = Block(256, 256)

        self.bn2 = nn.BatchNorm1d(num_features)

        self.block4 = Block(256, 512)

        self.bn3 = nn.BatchNorm1d(num_features)

        self.fc7 = nn.Linear(256 + 512, 256)

        self.adj = adj
        self.batch_adj = None

        self.relu = nn.ReLU()


    def load_batch_adj(self, x_in):
        bs = x_in.shape[0]

        adj_3d = np.zeros((bs, self.adj.shape[0], self.adj.shape[1]), dtype=float)

        for i in range(bs):
            adj_3d[i] = self.adj.cpu()

        adj_train = torch.FloatTensor(adj_3d)
        self.batch_adj = adj_train.to(x_in.device)

    def forward(self, x_in):
        self.load_batch_adj(x_in)

        x = self.fc1(x_in)

        x1 = self.relu(torch.bmm(self.batch_adj, x))
        x1 = self.block1(x1)

        x2 = self.relu(torch.bmm(self.batch_adj, x1))
        x2 = self.block2(x2)

        x3 = self.relu(torch.bmm(self.batch_adj, x2))

        x4 = torch.cat((x3, x1), 2)
        x4 = self.fc4(x4)
        x4 = self.bn1(x4)

        x5 = self.relu(torch.bmm(self.batch_adj, x4))
        x5 = self.block3(x5)
        x5 = self.bn2(x5)

        x6 = self.relu(torch.bmm(self.batch_adj, x5))
        x6 = self.block4(x6)

        x7 = self.relu(torch.bmm(self.batch_adj, x6))
        x7 = torch.cat((x7, x4), 2)
        x7 = self.bn3(self.fc7(x7))

        x = torch.cat((x7, x4, x1), 2)

        x = x.view(x.size(0), x.size(1) * x.size(2))

        return x

    def get_local_importance(self, x_in):
        x = self.forward(x_in)

        x = x.view(x.size(0), x.size(1))

        return x.cpu().data.numpy()