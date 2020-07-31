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

dataset = SpotifyRecommenderDataset()

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

model = Autoencoder()
model = train_autoencoder(model, train_dataloader)
with torch.no_grad():
    encodings = model.encode(dataset.model_input_tensor)
dataset.add_encoding_columns(encodings)

webapp = WebApp(dataset)

webapp.run()

a = 10