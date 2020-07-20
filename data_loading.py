import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from typing import Tuple, Union, List
from ast import literal_eval
from itertools import chain
from tqdm import tqdm


class SpotifyRecommenderDataset(Dataset):
    COLUMNS_TO_DROP = ['explicit', 'id', 'release_date', 'name']
    NUMERIC_COLUMNS = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key",
                            "liveness", "loudness", "mode", "popularity", "speechiness", "tempo", "valence"]
    DISTINCT_ARTISTS_COUNT = 27621
    DISTINCT_GENRES_COUNT = 17491
    NUMERIC_FIELDS_COUNT = 14

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
                 data_by_genres_path='data/data_by_genres.csv', pickle_path='data/dataset.pkl'):
        """Because the dataset is stored once it is created, only the first initialization takes long."""

        self.df_w_genres = pd.read_csv(data_w_genres_path)
        self.df_by_genres = pd.read_csv(data_by_genres_path)
        self._convert_string_column_to_list_type(self.df_w_genres, 'genres')

        if os.path.exists(pickle_path):
            self.df = pd.read_pickle(pickle_path)
        else:
            self.df = pd.read_csv(data_path)
            self._convert_string_column_to_list_type(self.df, 'artists')
            self._drop_unnecessary_columns()
            self.df['genres'] = self._genres_column()
            self._numerize_genres_and_artists_columns()
            self._normalize_numeric_columns()
            self.df.to_pickle(pickle_path)

    def _genres_column(self) -> pd.Series:
        """All unique genres of a song. The genres of a song are the genres of all artists of the song."""

        genres_column = []

        for i in tqdm(range(len(self.df.index)), desc="Collecting genres"):
            artists_of_song = self.df.loc[i, 'artists']
            bool_index = self.df_w_genres['artists'].isin(artists_of_song)
            genres_of_song = self.df_w_genres.loc[bool_index, 'genres']
            genres_of_song = list(set(chain.from_iterable(genres_of_song)))
            genres_column.append(genres_of_song)

        return pd.Series(genres_column)

    def _numerize_genres_and_artists_columns(self):
        distinct_genres = list(self.df_by_genres['genres'])
        genre_for_songs_without_genres = len(distinct_genres)
        numerized_genres_column = []

        for i in tqdm(range(len(self.df.index)), desc="Numerizing genres"):
            genres = self.df.loc[i, 'genres']
            if genres:
                numerized_genres = [distinct_genres.index(genre) for genre in genres]
            else:
                numerized_genres = [genre_for_songs_without_genres]
                genre_for_songs_without_genres += 1

            numerized_genres_column.append(numerized_genres)

        #numerized_genres_column = torch.transpose(torch.nn.utils.rnn.pad_sequence(numerized_genres_column), 0, 1).tolist()

        self.df['genres'] = numerized_genres_column

        distinct_artists = list(self.df_w_genres['artists'])
        artist_index_for_songs_without_artist = len(distinct_artists)
        numerized_artists_column = []
        for i in tqdm(range(len(self.df.index)), desc="Numerizing artists"):
            artists = self.df.loc[i, 'artists']
            numerized_artists = [distinct_artists.index(artist) for artist in artists if artist in distinct_artists]

            if not numerized_artists:
                numerized_artists = [artist_index_for_songs_without_artist]
                artist_index_for_songs_without_artist += 1

            numerized_artists_column.append(numerized_artists)

        #numerized_artists_column = torch.transpose(torch.nn.utils.rnn.pad_sequence(numerized_artists_column), 0, 1).tolist()

        self.df['artists'] = numerized_artists_column

    def _convert_string_column_to_list_type(self, dataframe: pd.DataFrame, column: str):
        string_column = dataframe[column]
        list_column = [literal_eval(string_element) for string_element in string_column]
        dataframe[column] = list_column

    def _drop_unnecessary_columns(self):
        self.df.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def _normalize_numeric_columns(self):
        for col in self.NUMERIC_COLUMNS:
            max_value = self.df[col].max()
            min_value = self.df[col].min()
            self.df[col] = (self.df[col] - min_value) / (max_value - min_value)

    def __len__(self):
        return len(self.df.index)

    
    ReturnType = Tuple[List, torch.tensor, List]

    def __getitem__(self, idx: Union[int, slice, list]) -> ReturnType:
        numeric_fields = self.df.loc[idx, self.df.columns.difference(['artists', 'genres'])]
        numeric_fields_tensor = torch.tensor(numeric_fields.values.astype('float32')).squeeze()

        artists = list(self.df.loc[idx, 'artists'])
        genres = list(self.df.loc[idx, 'genres'])

        return artists, numeric_fields_tensor, genres

def SpotifyRecommenderDataLoader(*args, **kwargs):
    def own_collate_fn(batch: List[SpotifyRecommenderDataset.ReturnType]) -> SpotifyRecommenderDataset.ReturnType:
        all_artists = [artists for artists, numeric_fields, genres in batch]
        numeric_fields = torch.stack([numeric_fields for artists, numeric_fields, genres in batch])
        all_genres = [genres for artists, numeric_fields, genres in batch]

        return all_artists, numeric_fields, all_genres

    return torch.utils.data.DataLoader(*args, **kwargs, collate_fn=own_collate_fn)