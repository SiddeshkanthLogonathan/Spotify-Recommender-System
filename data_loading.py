from typing import Tuple, Union
from pandas import read_csv, DataFrame, Series, read_pickle
from torch.utils.data import Dataset
from torch import tensor
from ast import literal_eval
from itertools import chain
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from torch import tensor
import os

class SpotifyRecommenderDataset(Dataset):
    COLUMNS_TO_DROP = ['explicit', 'id', 'release_date', 'name']
    COLUMNS_TO_NORMALIZE = ['key', 'loudness', 'popularity', 'tempo', 'speechiness', 'year', 'duration_ms']

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
                       data_by_genres_path='data/data_by_genres.csv', pickle_path='data/dataset.pkl'):
        """Because the dataset is stored once it is created, only the first initialization takes long."""

        self.df_w_genres = read_csv(data_w_genres_path)
        self.df_by_genres = read_csv(data_by_genres_path)
        self._convert_string_column_to_list_type(self.df_w_genres, 'genres')

        if os.path.exists(pickle_path):        
            self.df = read_pickle(pickle_path)
        else:
            self.df = read_csv(data_path)
            self._convert_string_column_to_list_type(self.df, 'artists')
            self._drop_columns()
            self.df['genres'] = self._genres_column()
            self._normalize_columns()
            self.df.to_pickle(pickle_path)

        self._setup_mlbs()

    def _genres_column(self) -> Series:
        """All unique genres of a song. The genres of a song are the genres of all artists of the song."""
        
        genres_column = []

        for i in tqdm(range(len(self.df.index)), desc="Collecting genres"):
            artists_of_song = self.df.loc[i, 'artists']
            bool_index = self.df_w_genres['artists'].isin(artists_of_song)
            genres_of_song = self.df_w_genres.loc[bool_index, 'genres']
            genres_of_song = list(set(chain.from_iterable(genres_of_song)))
            genres_column.append(genres_of_song)

        return Series(genres_column)

    def _setup_mlbs(self):
        """For one-hot/multiple-hot vectors. It would be insane to store these."""

        artists = set(self.df_w_genres['artists'])
        self.artists_mlb = MultiLabelBinarizer().fit([artists])
        
        genres = set(self.df_by_genres['genres'].drop(index=1))
        self.genres_mlb = MultiLabelBinarizer().fit([genres])
        
    def _convert_string_column_to_list_type(self, dataframe: DataFrame, column: str):
        string_column = dataframe[column]
        list_column = [literal_eval(string_element) for string_element in string_column]
        dataframe[column] = list_column

    def _drop_columns(self):
        self.df.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def _normalize_columns(self):
        for col in self.COLUMNS_TO_NORMALIZE:
            max_value = self.df[col].max()
            min_value = self.df[col].min()
            self.df[col] = (self.df[col] - min_value) / (max_value - min_value)

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx: Union[int, slice, list]) -> Tuple[tensor, tensor, tensor]:      
        numeric_fields = self.df.loc[idx, self.df.columns.difference(['artists', 'genres'])]
        numeric_fields_tensor = tensor(numeric_fields.values.astype('float64'))
        
        artists = list(self.df.loc[idx, 'artists'])
        genres = list(self.df.loc[idx, 'genres'])
        if type(idx) == int:
            artists = [artists]
            genres = [genres]
        
        artists_tensor = tensor(self.artists_mlb.transform(artists))
        genres_tensor = tensor(self.genres_mlb.transform(genres))

        return (artists_tensor, numeric_fields_tensor, genres_tensor)