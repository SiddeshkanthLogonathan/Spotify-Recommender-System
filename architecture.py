import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from itertools import chain
from typing import List
from data_loading import SpotifyRecommenderDataset
from gensim.models import Word2Vec


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
        # self.artist_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_ARTISTS_COUNT, 3)
        # self.genre_embedder = nn.Embedding(SpotifyRecommenderDataset.DISTINCT_GENRES_COUNT, 3)

        self.encoding_module = nn.Sequential(
            nn.Linear(14, 10),
            nn.Tanh(),
            nn.Linear(10, 6),
            nn.Tanh(),
            nn.Linear(6, 3)
        )
        self.decoding_module = nn.Sequential(
            nn.Tanh(),
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 10),
            nn.Tanh(),
            nn.Linear(10, 14)
        )

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

class GenreEmbedder(nn.Module):
    EMBEDDING_DIM = 3

    def __init__(self):
        super(GenreEmbedder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):
        self.embed = torch.nn.Embedding(SpotifyRecommenderDataset.DISTINCT_GENRES_COUNT, self.EMBEDDING_DIM)
        self.fully_connected = nn.Sequential(
            torch.nn.Linear(3, 20),
            torch.nn.Sigmoid(),
            torch.nn.Linear(20, 80),
            torch.nn.Sigmoid(),
            torch.nn.Linear(80, 768),
            torch.nn.Linear(768, SpotifyRecommenderDataset.DISTINCT_GENRES_COUNT),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input: torch.tensor):
        embeddings = self.embed(input)
        result = self.fully_connected(embeddings)
        a = 10
        return result



