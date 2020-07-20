import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from itertools import chain


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.01)


class Autoencoder(nn.Module):
    # LAYER_SIZES = [14, 9, 3]
    LAYER_SIZES = [20, 10, 6, 3]
    LAYER_COUNT = len(LAYER_SIZES)

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):

        self.artist_embedder = nn.Embedding(27621, 3)
        self.genre_embedder = nn.Embedding(2664, 3)

        # encoding architecture
        encoding_linear_layers = [nn.Linear(self.LAYER_SIZES[i], self.LAYER_SIZES[i+1]) for i in range(self.LAYER_COUNT - 1)]
        encoding_activation_layers = [torch.nn.ReLU() for i in range(len(encoding_linear_layers))]
        
        reverse_layer_sizes = list(reversed(self.LAYER_SIZES))
        decoding_linear_layers = [nn.Linear(reverse_layer_sizes[i], reverse_layer_sizes[i+1]) for i in range(self.LAYER_COUNT - 1)]
        decoding_activation_layers = [torch.nn.ReLU() for i in range(len(encoding_linear_layers))]

        encoding_layers = list(chain.from_iterable(zip(encoding_linear_layers, encoding_activation_layers)))
        decoding_layers = list(chain.from_iterable(zip(decoding_linear_layers, decoding_activation_layers)))

        self.encoding_module = nn.Sequential(*encoding_layers)
        self.decoding_module = nn.Sequential(*decoding_layers)



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
            artists_embedding = self.artist_embedder(indices)

            indices = list(filter((0).__ne__, x[i][2]))
            indices = list(filter(lambda idx: idx < 2664, indices))
            if len(indices) == 0:
                indices = [0]
            indices = torch.LongTensor(indices)
            genres_embedding = self.genre_embedder(indices)

            artists_embedding = torch.sum(artists_embedding, dim=0) / artists_embedding.shape[0]
            genres_embedding = torch.sum(genres_embedding, dim=0) / genres_embedding.shape[0]

            temp_x = torch.cat((artists_embedding.double(), x[i][1].double(), genres_embedding.double()), 0)
            if out is None:
                out = temp_x.unsqueeze(0)
            else:
                out = torch.cat((out, temp_x.unsqueeze(0)), dim=0)

        return out

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoding_module(x)

    def decode(self, x):
        return self.decoding_module(x)