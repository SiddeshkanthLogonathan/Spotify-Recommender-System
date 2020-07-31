import torch
import torch.nn as nn


def init_weights(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.build_architecture()
        self.apply(init_weights)

    def build_architecture(self):
        self.encode = nn.Sequential(
            nn.Linear(92, 70),
            nn.Tanh(),
            nn.Linear(70, 40),
            nn.Tanh(),
            nn.Linear(40, 16)
        )
        self.decode = nn.Sequential(
            nn.Tanh(),
            nn.Linear(16, 40),
            nn.Tanh(),
            nn.Linear(40, 70),
            nn.Tanh(),
            nn.Linear(70, 92)
        )

    def forward(self, batch: torch.tensor):
        return self.decode(self.encode(batch))
