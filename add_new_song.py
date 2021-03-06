from data_loading import SpotifyRecommenderDataset
import torch
import json
import sys
import os
import shutil

mandatory_keys = [
    "name", "accousticness", "danceability",
    "duration_ms", "energy", "instrumentalness", "key",
    "liveness", "loudness", "mode", "popularity",
    "speechiness", "tempo", "artists", "genres", "year", "valence"]

model_store_path = "./model.pth"


def add_song_to_csv(song, data_path='data/data.csv', data_w_genres_path='data/data_w_genres.csv'):
    with open(data_path, "a") as file:
        file.write(
            song["accousticness"] + "," +
            "\""+ str(song["artists"]) + "\"," +
            song["danceability"] + "," +
            song["duration_ms"] + "," +
            song["energy"] + "," +
            "0," +  # explicit field irrelevant
            "0 ," +  # id field irrelevant
            song["instrumentalness"] + "," +
            song["key"] + "," +
            song["liveness"] + "," +
            song["loudness"] + "," +
            song["mode"] + "," +
            song["name"] + "," +
            song["popularity"] + "," +
            "0," +  # release date irrelevant
            song["speechiness"] + "," +
            song["tempo"] + "," +
            song["valence"] + "," +
            song["year"] + "\n"
        )

    with open(data_w_genres_path, "a") as file:
        for artist in song["artists"]:
            file.write(
                artist + "," +
                song["accousticness"] + "," +
                song["danceability"] + "," +
                song["duration_ms"] + "," +
                song["energy"] + "," +
                song["instrumentalness"] + "," +
                song["liveness"] + "," +
                song["loudness"] + "," +
                song["speechiness"] + "," +
                song["tempo"] + "," +
                song["valence"] + ", " +
                song["popularity"] + "," +
                song["key"] + "," +
                song["mode"] + "," +
                "0," +  # count field irrelevant
                str(song["genres"]) + "\n"
            )


def add_song_encoding(song, model=None):
    if model is None:
        if os.path.exists(model_store_path):
            model = torch.load(model_store_path)
        else:
            raise ValueError("No Model provided, cannot add encoding!")
    model.eval()
    try:
        if os.path.exists(SpotifyRecommenderDataset.dir_for_storing):
            shutil.rmtree(SpotifyRecommenderDataset.dir_for_storing)
    except:
        print("could not delete dataset files file")
    dataset = SpotifyRecommenderDataset()
    with torch.no_grad():
        encodings = model.encode(dataset.model_input_tensor)
    dataset.add_encoding_columns(encodings)
    torch.save(model, model_store_path)


def add_new_song(song, model=None):
    print("adding new song art ", song["artists"], " type is: ", type(song["artists"]))

    print("adding new song!")
    add_song_to_csv(song)
    add_song_encoding(song, model)
    print("song successfully added!")


def main():
    try:
        song = json.load(sys.stdin)
    except:
        print("no valid json input provided!")
        sys.exit(1)

    corrupted = False
    for key in mandatory_keys:
        if not key in song:
            print("ERROR: key ", key, " not present in song definition!")
            corrupted = True
        elif not type(song[key]) is list:
                song[key] = str(song[key])
    if corrupted:
        raise ValueError("Some song entries were not present!")
    else:
        add_new_song(song)


if __name__ == "__main__":
    main()
