import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from itertools import chain
from typing import List
from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.01)


class Autoencoder(nn.Module):
    LAYER_SIZES = [20, 10, 6, 3]
    LAYER_COUNT = len(LAYER_SIZES)

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):

        self.artist_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_ARTISTS_COUNT, 3)
        self.genre_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_GENRES_COUNT, 3)

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

    def _embed_artists_or_genres(self, artist_or_genres_lists: List[List[int]], embedder: nn.Embedding) -> torch.tensor:
        avg_embeddings = []
        for artist_or_genre_list in artist_or_genres_lists:
            input_tensor = torch.tensor(artist_or_genre_list)
            embeddings = embedder(input_tensor)
            avg_embedding = torch.mean(embeddings, dim=0)
            avg_embeddings.append(avg_embedding)

        return torch.stack(avg_embeddings)

    def _embed_artists(self, artist_lists: List[List[int]]) -> torch.tensor:
        return self._embed_artists_or_genres(artist_lists, self.artist_embedder)

    def _embed_genres(self, genre_lists: List[List[int]]) -> torch.tensor:
        return self._embed_artists_or_genres(genre_lists, self.genre_embedder)

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

    def forward(self, x: SpotifyRecommenderDataset.ReturnType) -> torch.tensor:
        artists, numeric_fields_tensor, genres = x
        artist_embeddings = self._embed_artists(artists)
        genre_embeddings = self._embed_genres(genres)
        
        encoding_input = torch.cat([artist_embeddings, numeric_fields_tensor, genre_embeddings], dim=1)

        return self.decode(self.encode(encoding_input))

    def encode(self, x: torch.tensor):
        return self.encoding_module(x)

    def decode(self, x: torch.tensor):
        return self.decoding_module(x)

dataset = SpotifyRecommenderDataset()
data_loader = SpotifyRecommenderDataLoader(dataset, batch_size=10, shuffle=True)
autoencoder = Autoencoder()


for x in data_loader:
    print(autoencoder(x).shape)
    break