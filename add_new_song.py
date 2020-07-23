from data_loading import SpotifyRecommenderDataset
import torch

class Song():
    def __init__(self):
        self.name = ""
        self.accousticness = 0
        self.danceability = 0
        self.duration_ms = 0
        self.energy = 0
        self.instrumentalness = 0
        self.key = 0
        self.liveness = 0
        self.loudness = 0
        self.mode = 0
        self.popularity = 0
        self.speechiness = 0
        self.tempo = 0
        self.artist = []
        self.genre = []


def add_song_to_csv(song, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
        data_by_genres_path='data/data_by_genres.csv'):
    
    with open(data_path, "a") as file:
        file.write(
            song.accousticness + "," +
            song.artist + "," +
            song.danceability + "," +
            song.duration_ms + "," +
            song.energy + "," +
            "0," + #explicit field irrelevant
            "0 ," + #id field irrelevant
            song.instrumentalness + "," +
            song.key + "," +
            song.liveness + "," +
            song.loudness + "," +
            song.mode + "," +
            song.name + "," +
            song.popularity + "," +
            "0," + # release date irrelevant
            song.speechiness + "," +
            song.tempo + "," +
            "0," +# valence field irrelevant
            "0" # year irrelevant?
            )
    with open(data_by_genres_path, "a") as file:
        file.write(
            song.genre + ", " +
            song.accousticness + "," +
            song.danceability + "," +
            song.duration_ms + "," +
            song.energy + "," +
            song.instrumentalness + "," +
            song.liveness + "," +
            song.loudness + "," +
            song.speechiness + "," +
            song.tempo + "," +
            "0," +# valence field irrelevant
            song.popularity + "," +
            song.key + "," +
            song.mode
            )

    with open(data_w_genres_path, "a") as file:
        file.write(
            song.artist + "," +
            song.accousticness + "," +
            song.danceability + "," +
            song.duration_ms + "," +
            song.energy + "," +
            song.instrumentalness + "," +
            song.liveness + "," +
            song.loudness + "," +
            song.speechiness + "," +
            song.tempo + "," +
            "0," +# valence field irrelevant
            song.popularity + "," +
            song.key + "," +
            song.mode + "," +
            "0," + #count field irrelevant
            song.genres
            )


def add_song_encoding(song, net = torch.load('./model.pth')):
    net.eval()
    encoded_song = 0


def add_new_song(song):
    add_song_to_csv(song)
    add_song_encoding(song)