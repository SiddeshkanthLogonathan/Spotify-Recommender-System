import torch
import torch.nn as nn
import pandas as pd
from architecture import Autoencoder
import typing
from architecture import Autoencoder
from torch.utils.data import DataLoader
import csv
from data_loading import SpotifyRecommenderDataset


def test_and_save_encodings_to_csv():
    net = torch.load('./model.pth')
    net.eval()
    dataset = SpotifyRecommenderDataset()
    criterion = nn.MSELoss()

    encodings = [["name", "id", "enc1", "enc2", "enc3"]]
    original_data = pd.read_csv("data/data.csv")
    song_names = original_data["name"].to_list()
    song_artists = original_data["artists"].to_list()

    with torch.no_grad():
        running_loss = 0
        print("len dataset: ", len(dataset))
        for index, value in enumerate(dataset):
            if(index == len(dataset) - 1):
                break
            if(index % 2000 == 0):
                print("progress: ", index / len(dataset) * 100, "%")
            value = value.training_label
            # value = value.double()
            encoded_value = net.encode(value)
            value_to_save = [song_names[index-1], song_artists[index-1]]
            value_to_save += encoded_value.tolist()
            encodings.append(value_to_save)
            output = net.decode(encoded_value)
            loss = criterion(output, value)
            running_loss += loss
        print("Got overall testing loss: ", running_loss / len(dataset))
        print("encodings shape: ", len(encodings),", ", len(encodings[0]))

    with open("data/encodings.csv","w+") as file:
        csvWriter = csv.writer(file,delimiter=',')
        csvWriter.writerows(encodings)

def get_encodings(model: Autoencoder) -> torch.tensor:
    dataset = SpotifyRecommenderDataset()
    batch = dataset[0:len(dataset)]
    return model.encode(batch.training_label)

def main():
    model = torch.load('./model.pth')
    encodings = get_encodings(model)

if __name__ == "__main__":
    main()