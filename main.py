from data_loading import SpotifyRecommenderDataset
import visualization
from architecture import Autoencoder
from train import train_autoencoder
from webapp import WebApp
import torch
import os
import argparse
import shutil
from model_testing import test

model_store_path = "./model.pth"
dataframe_store_path = "data/spotify_recommender_dataset"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="force retraining of the model", action="store_true")
parser.add_argument("-v", "--verbose", help="verbose mode", action="store_true")
parser.add_argument("-c", "--clean", help="clean all cached dataframe files in \
    data/spotify_recommender_dataset/ and produce new ones (might take some time)", action="store_true")
parser.add_argument("-n", "--num-epochs", help="the number of epochs you want the model to \
    train. Only useful with -t option.", type=int, default=10)
args = parser.parse_args()

if args.clean and os.path.exists(dataframe_store_path):
    shutil.rmtree(dataframe_store_path)

trainModel = args.train
if not os.path.exists(model_store_path) or not os.path.exists(dataframe_store_path):
    trainModel = True

dataset = SpotifyRecommenderDataset()
if trainModel:
    print("starting to train model...")
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    model = Autoencoder()
    model = train_autoencoder(model, train_dataloader, verbose=args.verbose, num_epochs=args.num_epochs)
    with torch.no_grad():
        print("Evaluation loss:", test(model, dataset.model_input_tensor))
        encodings = model.encode(dataset.model_input_tensor)
    dataset.add_encoding_tensor(encodings)
    torch.save(model, model_store_path)

knn = visualization.KNN(dataset)
webapp = WebApp(dataset, knn)

webapp.run()
