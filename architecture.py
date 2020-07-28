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
    ENCODING_LAYER_SIZES = [14, 10, 6, 3]
    DECODING_LAYER_SIZES = [3, 6, 10, 14]
    ENCODING_LAYER_COUNT = len(ENCODING_LAYER_SIZES)
    DECODING_LAYER_COUNT = len(DECODING_LAYER_SIZES)

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):
        # self.artist_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_ARTISTS_COUNT, 3)
        # self.genre_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_GENRES_COUNT, 3)

        # encoding architecture
        encoding_linear_layers = [nn.Linear(self.ENCODING_LAYER_SIZES[i], self.ENCODING_LAYER_SIZES[i + 1]) for i in
                                  range(self.ENCODING_LAYER_COUNT - 1)]
        encoding_activation_layers = [torch.nn.ReLU() for i in range(len(encoding_linear_layers))]

        decoding_linear_layers = [nn.Linear(self.DECODING_LAYER_SIZES[i], self.DECODING_LAYER_SIZES[i + 1]) for i in
                                  range(self.DECODING_LAYER_COUNT - 1)]
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

    def forward(self, batch: SpotifyRecommenderDataset.ReturnType) -> torch.tensor:
        # artists, numeric_fields_tensor, genres = batch.artists, batch.numeric_fields, batch.genres
        # artist_embeddings = self._embed_artists(artists)
        # genre_embeddings = self._embed_genres(genres)

        # encoding_input = torch.cat([artist_embeddings, numeric_fields_tensor, genre_embeddings], dim=1)

        return self.decode(self.encode(batch.training_label))

    def encode(self, x: torch.tensor):
        return self.encoding_module(x)

    def decode(self, x: torch.tensor):
        return self.decoding_module(x)