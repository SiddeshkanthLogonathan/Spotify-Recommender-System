import pandas as pd

# The Data Frames are named according to its csv-filename in the format of pandas-DataFrame
data = pd.read_csv('./data/data.csv')
data_by_artist = pd.read_csv('./data/data_by_artist.csv')
data_by_genres = pd.read_csv('./data/data_by_genres.csv')
data_by_year = pd.read_csv('./data/data_by_year.csv')
data_w_year = pd.read_csv('./data/data_w_genres.csv')


