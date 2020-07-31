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
            nn.Linear(92, 82),
            nn.Tanh(),
            nn.Linear(82, 72),
            nn.Tanh(),
            nn.Linear(72, 62),
            nn.Tanh(),
            nn.Linear(62, 52),
            nn.Tanh(),
            nn.Linear(52, 42),
            nn.Tanh(),
            nn.Linear(42, 32),
            nn.Tanh(),
            nn.Linear(32, 22),
            nn.Tanh(),
            nn.Linear(22, 12)
        )
        self.decode = nn.Sequential(
            nn.Tanh(),
            nn.Linear(12, 22),
            nn.Tanh(),
            nn.Linear(22, 32),
            nn.Tanh(),
            nn.Linear(32, 42),
            nn.Tanh(),
            nn.Linear(42, 52),
            nn.Tanh(),
            nn.Linear(52, 62),
            nn.Tanh(),
            nn.Linear(62, 72),
            nn.Tanh(),
            nn.Linear(72, 82),
            nn.Tanh(),
            nn.Linear(82, 92)
        )

    def forward(self, batch: torch.tensor):
        return self.decode(self.encode(batch))