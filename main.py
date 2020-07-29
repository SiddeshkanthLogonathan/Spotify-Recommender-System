from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
from train import train
from model_testing import test
from webapp import WebApp
import torch

dataset = SpotifyRecommenderDataset()
dataloader = SpotifyRecommenderDataLoader(dataset, batch_size = 256, shuffle=True)
model = Autoencoder()
model = train(model, dataloader)
torch.save(model, 'model.pth')

loss = test(model)
print("Loss: ", loss)

webapp = WebApp()
webapp.run()