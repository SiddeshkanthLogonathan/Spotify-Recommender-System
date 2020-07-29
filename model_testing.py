import torch
import torch.nn as nn
from architecture import Autoencoder
from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader

def test(model: Autoencoder) -> int:
    model.eval()
    dataset = SpotifyRecommenderDataset()
    dataloader = SpotifyRecommenderDataLoader(dataset, batch_size = len(dataset))
    criterion = nn.MSELoss()
    batch = next(iter(dataloader))
    with torch.no_grad():
        outputs = model(batch)
        return criterion(outputs, batch.training_label)