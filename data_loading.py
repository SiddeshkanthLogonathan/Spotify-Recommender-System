import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from typing import Tuple, Union, List
from ast import literal_eval
from itertools import chain
from tqdm import tqdm
from statistics import mean
from collections import namedtuple
import numpy as np
from gensim.models import Word2Vec
import pickle


class SpotifyRecommenderDataset(Dataset):

    def __new__(cls):

        raw_data_path = 'data/data.csv'
        data_w_genres_path = 'data/data_w_genres.csv'
        pickle_path = 'data/spotify_recommender_dataset.pkl'

        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as file:
                instance = pickle.load(pickle_path)
        else:
            instance = super(SpotifyRecommenderDataset, cls).__new__(cls)

            instance.pickle_path = pickle_path
            instance.data_w_genres_path = data_w_genres_path
            instance.raw_data_path = raw_data_path

            instance.df = pd.read_csv(raw_data_path)
            instance._convert_string_column_to_list_type(instance.df, 'artists')
            instance._join_genres_column_into_main_df()
            instance.df.to_pickle(pickle_path)
            instance.model_input_tensor = instance._create_model_input_tensor()

            pickle.dump(instance, open(pickle_path, 'wb'))

        return instance
    """
    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
                 data_by_genres_path='data/data_by_genres.csv', pickle_path='data/dataset.pkl'):

        self.pickle_path = pickle_path
        self.data_w_genres_path = data_w_genres_path

        if os.path.exists(pickle_path):
            self.df = pd.read_pickle(pickle_path)
            self.model_input_tensor = self._create_model_input_tensor()
        else:
            self.df = pd.read_csv(data_path)
            self._convert_string_column_to_list_type(self.df, 'artists')
            self._join_genres_column_into_main_df()
            self.df.to_pickle(pickle_path)
            self.model_input_tensor = self._create_model_input_tensor()
    """

    def _word2vec_objects(self):
        genre_corpus = self.df['genres'].tolist()
        artist_corpus = self.df['artists'].tolist()

        max_length_of_genres_lists = max(len(genres) for genres in genre_corpus)
        max_length_of_artists_lists = max(len(artists) for artists in artist_corpus)

        genre_word2vec = Word2Vec(genre_corpus, size=5, min_count=1, window=max_length_of_genres_lists+1)
        artist_word2vec = Word2Vec(artist_corpus, size=5, min_count=1, window=max_length_of_artists_lists+1)

        return genre_word2vec, artist_word2vec

    def _embedding_columns(self):
        genre_word2vec, artist_word2vec = self._word2vec_objects()

        artist_embedding_column = []
        for artist_list in self.df['artists']:
            artists_embeddings = artist_word2vec.wv[artist_list]
            mean_embedding = artists_embeddings.mean(axis=0)
            artist_embedding_column.append(mean_embedding)
        artist_embedding_column = np.stack(artist_embedding_column)

        genre_embedding_column = []
        for genre_list in self.df['genres']:
            genre_embeddings = genre_word2vec.wv[genre_list]
            mean_embedding = genre_embeddings.mean(axis=0)
            genre_embedding_column.append(mean_embedding)
        genre_embedding_column = np.stack(genre_embedding_column)

        return torch.tensor(genre_embedding_column), torch.tensor(artist_embedding_column)

    def _create_model_input_tensor(self):
        numeric_columns = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key",
                           "liveness", "loudness", "mode", "popularity", "speechiness", "tempo", "valence", 'year']

        numeric_df = self.df[numeric_columns]
        numeric_tensor = torch.tensor(numeric_df.values).float()
        normed_numeric_tensor = \
            (numeric_tensor - torch.mean(numeric_tensor, 0, keepdim=True)) / torch.std(numeric_tensor, 0, keepdim=True)

        genre_embedding_column, artist_embedding_column = self._embedding_columns()

        return torch.cat([normed_numeric_tensor, genre_embedding_column, artist_embedding_column], dim=1)

    def add_encoding_columns(self, encodings: Union[torch.tensor, np.array]):
        self.df['encoding_x'] = encodings[:, 0]
        self.df['encoding_y'] = encodings[:, 1]
        self.df['encoding_z'] = encodings[:, 2]
        self.df.to_pickle(self.pickle_path)

    def _join_genres_column_into_main_df(self) -> pd.Series:
        """All unique genres of a song. The genres of a song are the genres of all artists of the song."""

        df_w_genres = pd.read_csv(self.data_w_genres_path)
        self._convert_string_column_to_list_type(df_w_genres, 'genres')
        genres_column = []
        none_genre = -1

        for i in tqdm(range(len(self.df.index)), desc="Joining genres into main dataframe"):
            artists_of_song = self.df.loc[i, 'artists']
            bool_index = df_w_genres['artists'].isin(artists_of_song)
            genres_of_song = df_w_genres.loc[bool_index, 'genres']
            genres_of_song = list(set(chain.from_iterable(genres_of_song)))

            if not genres_of_song:
                genres_of_song = [str(none_genre)]
                none_genre -= 1

            genres_column.append(genres_of_song)

        self.df['genres'] = genres_column

    def _convert_string_column_to_list_type(self, dataframe: pd.DataFrame, column: str):
        string_column = dataframe[column]
        list_column = [literal_eval(string_element) for string_element in string_column]
        dataframe[column] = list_column

    def __len__(self):
        return len(self.model_input_tensor)

    def __getitem__(self, idx: Union[int, slice, list]):
        return self.model_input_tensor[idx]
