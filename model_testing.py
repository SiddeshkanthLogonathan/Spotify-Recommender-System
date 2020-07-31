import torch
import torch.nn as nn
from architecture import Autoencoder
from data_loading import SpotifyRecommenderDataset

def test(model: Autoencoder) -> int:
    model.eval()
    dataset = SpotifyRecommenderDataset()
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(dataset.model_input_tensor)
        return criterion(outputs, dataset.model_input_tensor)