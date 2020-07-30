from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
from train import train
from model_testing import test
from webapp import WebApp
import torch
from visualization import GeneralPurposeVisualizer

dataset = SpotifyRecommenderDataset()
dataloader = SpotifyRecommenderDataLoader(dataset, batch_size = 256, shuffle=True)
model = Autoencoder()
model = train(model, dataloader)
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