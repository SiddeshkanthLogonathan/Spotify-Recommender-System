from data_loading import SpotifyRecommenderDataset

"""
from architecture import Autoencoder, GenreEmbedder
from train import train_autoencoder, train_genre_embedder
from model_testing import test
from webapp import WebApp
import torch
from visualization import GeneralPurposeVisualizer
import pandas as pd
import plotly.express as px
from gensim.test.utils import common_texts, get_tmpfile
from itertools import chain
from gensim.models import Word2Vec
"""


"""
dataset = SpotifyRecommenderDataset()
dataloader = SpotifyRecommenderDataLoader(dataset, batch_size = 256, shuffle=True)
model = Autoencoder()
model = train_autoencoder(model, dataloader)
torch.save(model, 'model.pth')

loss = test(model)
print("Loss: ", loss)

dataloader = SpotifyRecommenderDataLoader(dataset, batch_size=len(dataset))
batch = next(iter(dataloader))

with torch.no_grad():
    encodings = model.encode(batch.training_label)
dataset.add_encoding_columns(encodings)
#GeneralPurposeVisualizer.visualize_encodings(encodings)

webapp = WebApp()
webapp.run()
"""


dataset = SpotifyRecommenderDataset()