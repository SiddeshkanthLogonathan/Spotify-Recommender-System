from pandas import read_csv, DataFrame
from torch.utils.data import Dataset, DataLoader
from torch import tensor, set_printoptions
from ast import literal_eval
import typing


class SpotifyRecommenderDataset(Dataset):
    COLUMNS_TO_DROP = ['explicit', 'id', 'release_date', 'artists', 'name']
    COLUMNS_TO_NORMALIZE = ['key', 'loudness', 'popularity', 'tempo', 'speechiness', 'year']

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv'):
        self.df = read_csv(data_path)
        self.df_w_genres = read_csv(data_w_genres_path)
        self._convert_column_to_string_type(self.df, 'artists')
        self._convert_column_to_string_type(self.df_w_genres, 'genres')
        
        self._join_genres_into_songs()

        #self._drop_non_numeric_columns()
        #self._normalize_columns()
        #self.tensor = tensor(self.df.values)

    def _join_genres_into_songs(self):
        self.df['genres'] = [list()] * len(self.df)
        for i in range(len(self.df)):
            print(i)

            for artist in self.df.loc[i, 'artists']:
                genres = self.df_w_genres.loc[self.df_w_genres['artists'] == artist, 'genres']
                for genre in genres:
                    self.df.loc[i, 'genres'].append(genre)
            
            if i == 100:
                break

    def _convert_column_to_string_type(self, dataframe: DataFrame, column: str):
        string_column = dataframe[column]
        list_column = [literal_eval(string_element) for string_element in string_column]
        dataframe[column] = list_column

    def _drop_non_numeric_columns(self):
        self.df.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def _normalize_columns(self):
        for col in self.COLUMNS_TO_NORMALIZE:
            max_value = self.df[col].max()
            min_value = self.df[col].min()
            self.df[col] = (self.df[col] - min_value) / (max_value - min_value)

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


def main():
    dataset = SpotifyRecommenderDataset()
    print(dataset.df['genres'].head(42))
    

if __name__ == "__main__":
    main()