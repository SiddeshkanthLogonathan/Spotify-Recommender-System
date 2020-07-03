import pandas as pd
import matplotlib.pyplot as plt

# The Data Frames are named according to its csv-filename in the format of pandas-DataFrame
data = pd.read_csv('./data/data.csv')
# data_by_artist = pd.read_csv('./data/data_by_artist.csv')
# data_by_genres = pd.read_csv('./data/data_by_genres.csv')
# data_by_year = pd.read_csv('./data/data_by_year.csv')
# data_w_year = pd.read_csv('./data/data_w_genres.csv')


# Remove unnecessary attributes ['explicit', 'id', 'year', 'release_date']
columns_to_drop = ['explicit', 'id', 'release_date']
data.drop(columns_to_drop, axis=1, inplace=True)


def min_max_normalize(df, columns):
    for col in columns:
        max_value = data[col].max()
        min_value = data[col].min()
        data[col] = (data[col] - min_value) / (max_value - min_value)
    return df


columns_to_normalize = ['key', 'loudness', 'popularity', 'tempo', 'speechiness', 'year']
min_max_normalize(data, columns_to_normalize)

data['year'].plot.hist(bins=10)
plt.show()
