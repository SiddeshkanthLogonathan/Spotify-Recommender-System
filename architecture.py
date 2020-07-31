import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from data_loading import SpotifyRecommenderDataset


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):
        self.encode = nn.Sequential(
            nn.Linear(44, 30),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Tanh(),
            nn.Linear(10, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
        )
        self.decode = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 10),
            nn.Tanh(),
            nn.Linear(10, 20),
            nn.Tanh(),
            nn.Linear(20, 30),
            nn.Tanh(),
            nn.Linear(30, 44)
        )

    def forward(self, batch: torch.tensor):
        return self.decode(self.encode(batch))