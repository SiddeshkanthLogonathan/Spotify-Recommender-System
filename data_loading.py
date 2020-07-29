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


class SpotifyRecommenderDataset(Dataset):
    NUMERIC_COLUMNS = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key",
                       "liveness", "loudness", "mode", "popularity", "speechiness", "tempo", "valence", 'year']
    DISTINCT_ARTISTS_COUNT = 27621
    DISTINCT_GENRES_COUNT = 17492
    NUMERIC_FIELDS_COUNT = 14

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
                 data_by_genres_path='data/data_by_genres.csv', pickle_path='data/dataset.pkl'):
        """Because the dataset is stored once it is created, only the first initialization takes long."""

        # self.df_by_genres = pd.read_csv(data_by_genres_path)
        # self._convert_string_column_to_list_type(self.df_w_genres, 'genres')

        if os.path.exists(pickle_path):
            self.df = pd.read_pickle(pickle_path)
        else:
            self.df = pd.read_csv(data_path)
            self.df_w_genres = pd.read_csv(data_w_genres_path)
            self._convert_string_column_to_list_type(self.df, 'artists')
            self._convert_string_column_to_list_type(self.df_w_genres, 'genres')
            self.df['genres'] = self._genres_column()
            # self._convert_string_column_to_list_type(self.df, 'genres')
            # self._numerize_genres_and_artists_columns()
            # self._normalize_numeric_columns()
            self.df.to_pickle(pickle_path)

        self.numeric_fields_tensor = self._create_numeric_fields_tensor()

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

        self.df['artists'] = numerized_artists_column

    def _convert_string_column_to_list_type(self, dataframe: pd.DataFrame, column: str):
        string_column = dataframe[column]
        list_column = [literal_eval(string_element) for string_element in string_column]
        dataframe[column] = list_column

    def _drop_unnecessary_columns(self):
        self.df.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def _normalize_numeric_columns(self):
        for col in self.NUMERIC_COLUMNS:
            mean = self.df[col].mean()
            stddev = self.df[col].std()
            self.df[col] = (self.df[col] - mean) / stddev * 100

    def _create_numeric_fields_tensor(self):
        df = self.df[self.NUMERIC_COLUMNS]
        tensor = torch.tensor(df.values).float()
        normed = (tensor - torch.mean(tensor, 0, keepdim=True)) / torch.std(tensor, 0, keepdim=True)

        return normed

    def __len__(self):
        return len(self.df.index)

    # ReturnType = Tuple[Tuple[List, torch.tensor, List], torch.tensor]
    ReturnType = namedtuple('ReturnType', ['artists', 'numeric_fields', 'genres', 'training_label'])

    def means_and_stddevs(self):
        return pd.DataFrame.from_dict({'acousticness': {'mean': 0.49731333996592436, 'stddev': 0.7414460292277519},
                                       'danceability': {'mean': 0.5422472298076697, 'stddev': 0.6620246753807381},
                                       'duration_ms': {'mean': 231396.66697073847, 'stddev': 121328.43694803145},
                                       'energy': {'mean': 0.49269268454059656, 'stddev': 0.6924105929679893},
                                       'instrumentalness': {'mean': 0.16242483376250616, 'stddev': 0.31851769768554716},
                                       'key': {'mean': 5.203600602650721, 'stddev': 3.5478173346586153},
                                       'liveness': {'mean': 0.21080151704371575, 'stddev': 0.6644608742444775},
                                       'loudness': {'mean': -11.369326520162904, 'stddev': 5.66863282734366},
                                       'mode': {'mean': 0.7091033216412815, 'stddev': 0.4623579581374077},
                                       'popularity': {'mean': 31.596506509098614, 'stddev': 22.45944013950889},
                                       'speechiness': {'mean': 0.09454818145436569, 'stddev': 0.16828479657505896},
                                       'tempo': {'mean': 116.95061491560536, 'stddev': 30.728968654447534},
                                       'valence': {'mean': 0.5320732217160243, 'stddev': 0.2624244490834021},
                                       'year': {'mean': 1977.141775936345, 'stddev': 28.566289822240513}})

    def __getitem__(self, idx: Union[int, slice, list]) -> ReturnType:
        numeric_fields_tensor = self.numeric_fields_tensor[idx]

        # artists = list(self.df.loc[idx, 'artists'])
        # genres = list(self.df.loc[idx, 'genres'])

        # mean_normalized_artist = self._normalize_artist_index(mean(artists))
        # mean_normalized_genre = self._normalize_genre_index(mean(genres))

        # mean_normalized_artist_tensor = torch.tensor([mean_normalized_artist])
        # mean_normalized_genre_tensor = torch.tensor([mean_normalized_genre])

        # training_label_tensor = torch.cat(
        #     [mean_normalized_artist_tensor, numeric_fields_tensor, mean_normalized_genre_tensor]
        # )

        # return self.ReturnType(artists=artists, numeric_fields=numeric_fields_tensor, genres=genres,
        #                        training_label=training_label_tensor)


        return self.ReturnType(artists=None, numeric_fields=None, genres=None, training_label=numeric_fields_tensor)


def SpotifyRecommenderDataLoader(*args, **kwargs):
    def custom_collate_fn(batch: List[SpotifyRecommenderDataset.ReturnType]) -> SpotifyRecommenderDataset.ReturnType:
        # all_artists = [sample.artists for sample in batch]
        # all_numeric_fields = torch.stack([sample.numeric_fields for sample in batch])
        # all_genres = [sample.genres for sample in batch]
        all_training_labels = torch.stack([sample.training_label for sample in batch])

        # return SpotifyRecommenderDataset.ReturnType(artists=all_artists, numeric_fields=all_numeric_fields,
        #                                             genres=all_genres, training_label=all_training_labels)
        return SpotifyRecommenderDataset.ReturnType(artists=None, numeric_fields=None,
                                                    genres=None, training_label=all_training_labels)

    return torch.utils.data.DataLoader(*args, **kwargs, collate_fn=custom_collate_fn)
