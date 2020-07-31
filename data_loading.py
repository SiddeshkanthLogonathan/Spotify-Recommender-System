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
    raw_data_path = 'data/data.csv'
    raw_data_w_genres_path = 'data/data_w_genres.csv'

    dir_for_storing = 'data/spotify_recommender_dataset'

    df_path = os.path.join(dir_for_storing, 'df.pkl')
    model_input_tensor_path = os.path.join(dir_for_storing, 'model_input_tensor.pt')
    genre_word2vec_path = os.path.join(dir_for_storing, 'genre_word2vec.pkl')
    artist_word2vec_path = os.path.join(dir_for_storing, 'artist_word2vec.pkl')

    encodings_tensor_path = os.path.join(dir_for_storing, 'encodings_tensor.pt')

    ARTIST_EMBEDDING_DIM = 20
    GENRE_EMBEDDING_DIM = 30

    def __init__(self):
        if os.path.exists(self.dir_for_storing):
            self.df = pd.read_pickle(self.df_path)
            self.genre_word2vec = Word2Vec.load(self.genre_word2vec_path)
            self.artist_word2vec = Word2Vec.load(self.artist_word2vec_path)
            self.model_input_tensor = torch.load(self.model_input_tensor_path)
            try:
                print("Initializing with encodings tensor")
                self.encodings_tensor = torch.load(self.encodings_tensor_path)
            except FileNotFoundError:
                print("Encodings tensor not found. Initializing without it.")
        else:
            self.df = pd.read_csv(self.raw_data_path)
            self._convert_string_column_to_list_type(self.df, 'artists')
            self._join_genres_column_into_main_df()
            self.genre_word2vec, self.artist_word2vec = self._create_word2vec_objects()
            self.model_input_tensor = self._create_model_input_tensor()

            os.mkdir(self.dir_for_storing)
            self.df.to_pickle(self.df_path)
            self.genre_word2vec.save(self.genre_word2vec_path)
            self.artist_word2vec.save(self.artist_word2vec_path)
            torch.save(self.model_input_tensor, self.model_input_tensor_path)

    def _create_word2vec_objects(self):
        genre_corpus = self.df['genres'].tolist()
        artist_corpus = self.df['artists'].tolist()

        max_length_of_genres_lists = max(len(genres) for genres in genre_corpus)
        max_length_of_artists_lists = max(len(artists) for artists in artist_corpus)

        genre_word2vec = Word2Vec(genre_corpus, size=30, min_count=1, window=max_length_of_genres_lists + 1)
        artist_word2vec = Word2Vec(artist_corpus, size=20, min_count=1, window=max_length_of_artists_lists + 1)

        return genre_word2vec, artist_word2vec

    def _embedding_columns(self):
        artist_embedding_column = []
        for artist_list in self.df['artists']:
            artists_embeddings = self.artist_word2vec.wv[artist_list]
            mean_embedding = artists_embeddings.mean(axis=0)
            artist_embedding_column.append(mean_embedding)
        artist_embedding_column = np.stack(artist_embedding_column)

        genre_embedding_column = []
        for genre_list in self.df['genres']:
            genre_embeddings = self.genre_word2vec.wv[genre_list]
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

        result = torch.cat([normed_numeric_tensor, normed_numeric_tensor, normed_numeric_tensor, genre_embedding_column,
                            artist_embedding_column], dim=1)
        self.output_size = result.shape[1]

        return result

    def add_encoding_tensor(self, encodings: torch.tensor):
        self.encodings_tensor = encodings
        torch.save(encodings, self.encodings_tensor_path)

    def _join_genres_column_into_main_df(self):
        """All unique genres of a song. The genres of a song are the genres of all artists of the song."""

        df_w_genres = pd.read_csv(self.raw_data_w_genres_path)
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
