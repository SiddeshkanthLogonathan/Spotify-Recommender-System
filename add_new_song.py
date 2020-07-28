from data_loading import SpotifyRecommenderDataset
import torch
import json
import sys
import os

# class Song():
#     def __init__(self):
#         self.name = ""
#         self.id = "None"
#         self.accousticness = 0
#         self.danceability = 0
#         self.duration_ms = 0
#         self.energy = 0
#         self.instrumentalness = 0
#         self.key = 0
#         self.liveness = 0
#         self.loudness = 0
#         self.mode = 0
#         self.popularity = 0
#         self.speechiness = 0
#         self.tempo = 0
#         self.artist = []
#         self.genre = []


mandatory_keys = [
    "name", "accousticness", "danceability", 
    "duration_ms", "energy", "instrumentalness", "key", 
    "liveness", "loudness", "mode", "popularity", 
    "speechiness", "tempo", "artist", "genre"]

def add_song_to_csv(song, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv',
        data_by_genres_path='data/data_by_genres.csv'):
    
    with open(data_path, "a") as file:
        file.write(
            song["accousticness"] + "," +
            song["artist"] + "," +
            song["danceability"] + "," +
            song["duration_ms"] + "," +
            song["energy"] + "," +
            "0," + #explicit field irrelevant
            "0 ," + #id field irrelevant
            song["instrumentalness"] + "," +
            song["key"] + "," +
            song["liveness"] + "," +
            song["loudness"] + "," +
            song["mode"] + "," +
            song["name"] + "," +
            song["popularity"] + "," +
            "0," + # release date irrelevant
            song["speechiness"] + "," +
            song["tempo"] + "," +
            "0," +# valence field irrelevant
            "0\n" # year irrelevant?
            )
    # with open(data_by_genres_path, "a") as file:
    #     file.write(
    #         song.genre + ", " +
    #         song.accousticness + "," +
    #         song.danceability + "," +
    #         song.duration_ms + "," +
    #         song.energy + "," +
    #         song.instrumentalness + "," +
    #         song.liveness + "," +
    #         song.loudness + "," +
    #         song.speechiness + "," +
    #         song.tempo + "," +
    #         "0," +# valence field irrelevant
    #         song.popularity + "," +
    #         song.key + "," +
    #         song.mode
    #         )

    # with open(data_w_genres_path, "a") as file:
    #     file.write(
    #         song.artist + "," +
    #         song.accousticness + "," +
    #         song.danceability + "," +
    #         song.duration_ms + "," +
    #         song.energy + "," +
    #         song.instrumentalness + "," +
    #         song.liveness + "," +
    #         song.loudness + "," +
    #         song.speechiness + "," +
    #         song.tempo + "," +
    #         "0," +# valence field irrelevant
    #         song.popularity + "," +
    #         song.key + "," +
    #         song.mode + "," +
    #         "0," + #count field irrelevant
    #         song.genres
    #         )


def add_song_encoding(song, net = torch.load('./model.pth'), encoding_path = "data/encodings.csv"):
    net.eval()
    try:
        os.remove("data/dataset.pkl")
    except:
        print("could not delete pickle file")
    new_dataset = SpotifyRecommenderDataset()
    new_encodings = net(new_dataset[len(new_dataset) - 1]).tolist()
    values_to_write = [song["name"], song["artist"]]
    values_to_write += new_encodings
    with open(encoding_path, "a") as file:
        file.write(str(values_to_write)[1:-1] + '\n')


def add_new_song(song):
    print("adding new song!")
    add_song_to_csv(song)
    add_song_encoding(song)
    print("song successfully added!")

def main():
    try:
        song = json.load(sys.stdin)
    except:
        print("no valid json input provided!")
    
    corrupted = False
    for key in mandatory_keys:
        if not key in song:
            print("key ", key, " not present in song definition!")
            corrupted = True
        else:
            song[key] = str(song[key])
    if corrupted:
        raise ValueError()
    else:
        add_new_song(song)

if __name__ == "__main__":
    main()