from pandas import read_csv, DataFrame
from torch.utils.data import Dataset, DataLoader
from torch import tensor, set_printoptions
from ast import literal_eval


class SpotifyRecommenderDataset(Dataset):
    COLUMNS_TO_DROP = ['explicit', 'id', 'release_date', 'artists', 'name']
    COLUMNS_TO_NORMALIZE = ['key', 'loudness', 'popularity', 'tempo', 'speechiness', 'year', 'duration_ms']

    def __init__(self, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv'):
        self.df = read_csv(data_path)
        self.df_w_genres = read_csv(data_w_genres_path)
        self._convert_column_to_string_type(self.df, 'artists')
        self._convert_column_to_string_type(self.df_w_genres, 'genres')
        
        # self._join_genres_into_songs()

        # self._drop_non_numeric_columns()
        # self._normalize_columns()
        # self.tensor = tensor(self.df.values)

    def _join_genres_into_songs(self):
        # self.df['genres'] = [list()] * len(self.df)
        rows_for_genres = []

        for i in range(len(self.df)):
            genres_by_artist = []
            genres = []
            for artists in self.df.loc[i, 'artists']:

                genres_for_artist = (self.df_w_genres.loc[self.df_w_genres['artists'] == artists, 'genres']).tolist()
                genres_by_artist.extend(genres_for_artist)

            for genre_list in genres_by_artist:
                genres.extend(genre_list)
            unique_sorted_genres = sorted(list(set(genres)))
            # self.df.loc[i, 'genres'] = unique_sorted_genres
            rows_for_genres.append(unique_sorted_genres)

        self.df['genres'] = rows_for_genres
        # This was only executed once to store the dataset so that it could be read from now onwards.
        # self.df.to_csv('data_and_genres.csv')

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
    # set_option('display.max_colwidth', -1)
    # set_option('display.max_colwidth', None)
    # set_option('display.max_columns', 10)
    # print(dataset.df.head(10))



if __name__ == "__main__":
    main()