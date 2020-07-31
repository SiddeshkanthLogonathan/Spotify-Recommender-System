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
            nn.Linear(72, 40),
            nn.Tanh(),
            nn.Linear(40, 25)
        )
        self.decode = nn.Sequential(
            nn.Tanh(),
            nn.Linear(25, 40),
            nn.Tanh(),
            nn.Linear(40, 72)
        )

    def forward(self, batch: torch.tensor):
        return self.decode(self.encode(batch))