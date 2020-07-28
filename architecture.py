import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.01)


class Autoencoder(nn.Module):
    LAYER_SIZES = [14, 10, 6, 3]
    # LAYER_SIZES = [20, 10, 6, 3]

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):

        # self.emb_artists = nn.Embedding(27621, 3)
        # self.emb_genres = nn.Embedding(2664, 3)

        # encoding architecture
        self.enc1 = nn.Linear(
            in_features=self.LAYER_SIZES[0],
            out_features=self.LAYER_SIZES[1])
        self.enc2 = nn.Linear(
            in_features=self.LAYER_SIZES[1],
            out_features=self.LAYER_SIZES[2])
        self.enc3 = nn.Linear(
            in_features=self.LAYER_SIZES[2],
            out_features=self.LAYER_SIZES[3])

        # decoding architecture
        self.dec1 = nn.Linear(
            in_features=self.LAYER_SIZES[3],
            out_features=self.LAYER_SIZES[2])
        self.dec2 = nn.Linear(
            in_features=self.LAYER_SIZES[2],
            out_features=self.LAYER_SIZES[1])
        self.dec3 = nn.Linear(
            in_features=self.LAYER_SIZES[1],
            out_features=self.LAYER_SIZES[0])

    def transform(self, x):

        out = None

        for i in range(len(x)):

            indices = list(filter((0).__ne__, x[i][0]))
            indices = torch.LongTensor(indices)
            artists_embedding = self.emb_artists(indices)

            indices = list(filter((0).__ne__, x[i][2]))
            indices = list(filter(lambda idx: idx < 2664, indices))
            if len(indices) == 0:
                indices = [0]
            indices = torch.LongTensor(indices)
            genres_embedding = self.emb_genres(indices)

            artists_embedding = torch.sum(artists_embedding, dim=0) / artists_embedding.shape[0]
            genres_embedding = torch.sum(genres_embedding, dim=0) / genres_embedding.shape[0]

            temp_x = torch.cat((artists_embedding.double(), x[i][1].double(), genres_embedding.double()), 0)
            if out is None:
                out = temp_x.unsqueeze(0)
            else:
                out = torch.cat((out, temp_x.unsqueeze(0)), dim=0)

        return out

    def forward(self, x):
        x = self.encode(x.double())
        x = self.decode(x.double())

        return x

    def encode(self, x):

        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))

        return x