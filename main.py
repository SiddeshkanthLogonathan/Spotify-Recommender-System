from data_loading import SpotifyRecommenderDataset
import torch

from architecture import Autoencoder
from train import train_autoencoder
from model_testing import test
from webapp import WebApp
import torch
from visualization import GeneralPurposeVisualizer
import pandas as pd
import plotly.express as px
from gensim.test.utils import common_texts, get_tmpfile
from itertools import chain
from gensim.models import Word2Vec
import pickle
import argparse
import os

model_store_path = "./data/spotify_recommender_dataset/model.pth"

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--train", help="force retraining of the model", action="store_true")
parser.add_argument("-v", "--verbose", help="verbose mode", action="store_true")
parser.add_argument("-n", "--num-epochs", help="the number of epochs you want the model to \
    train. Only useful with -t option.", type=int, default=10)
args = parser.parse_args()
dataset = SpotifyRecommenderDataset()

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

trainModel = args.train
if not os.path.exists(model_store_path):
    trainModel = True

if trainModel:
    print("starting to train model...")
    model = Autoencoder()
    model = train_autoencoder(model, train_dataloader, args.verbose, args.num_epochs)
    with torch.no_grad():
        encodings = model.encode(dataset.model_input_tensor)
    dataset.add_encoding_columns(encodings)
    torch.save(model, model_store_path)

webapp = WebApp(dataset)

webapp.run()

a = 10