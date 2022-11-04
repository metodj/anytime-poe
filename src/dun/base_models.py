import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim, eps=1e-5, momentum=0.1))

    def forward(self, x):
        return x + self.block(x)


class MLP(nn.Module):
    def __init__(self, input_dim, width, output_dim, depth):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, width)
        self.output_layer = nn.Linear(width, self.output_dim)
        self.depth = depth
        self.width = width

        self.layers = nn.ModuleList()
        for _ in range(self.depth):
            self.layers.append(ResBlock(width))

    def forward(self, x):
        act_vec = torch.zeros(self.depth+1, x.shape[0], self.output_dim).type(x.type())
        x = self.input_layer(x)
        act_vec[0] = self.output_layer(x)
        for i in range(self.depth):
            x = self.layers[i](x)
            act_vec[i+1] = self.output_layer(x)
        return act_vec
