from pandas import read_csv, DataFrame
from torch.utils.data import Dataset, DataLoader
from torch import tensor, set_printoptions
from ast import literal_eval


class SpotifyRecommenderDataset(Dataset):
    COLUMNS_TO_DROP = ['explicit', 'id', 'release_date', 'artists', 'name']
    COLUMNS_TO_NORMALIZE = ['key', 'loudness', 'popularity', 'tempo', 'speechiness', 'year']

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv'):
        self.df = read_csv(data_path)
        self.df_w_genres = read_csv(data_w_genres_path)
        # self._convert_strings_to_lists()
        self._drop_non_numeric_columns()
        self._normalize_columns()
        self.tensor = tensor(self.df.values)

    def _drop_non_numeric_columns(self):
        self.df.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def _normalize_columns(self):
        for col in self.COLUMNS_TO_NORMALIZE:
            max_value = self.df[col].max()
            min_value = self.df[col].min()
            self.df[col] = (self.df[col] - min_value) / (max_value - min_value)

    def _convert_strings_to_lists(self):
        # TODO: Fix ValueError
        self.df["artists"] = literal_eval(self.df["artists"])
        self.df_w_genres["genres"] = literal_eval(self.df_w_genres["genres"])

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]
