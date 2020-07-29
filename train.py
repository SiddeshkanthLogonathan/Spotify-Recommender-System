from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
import torch

default_num_epochs = 10
default_criterion = torch.nn.MSELoss()
default_learning_rate = 0.01
default_decay_rate = 1.0

def train(model: Autoencoder, dataloader: torch.utils.data.DataLoader, num_epochs=default_num_epochs,
          criterion=default_criterion, verbose=True):

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                     amsgrad=False)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=default_decay_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.training_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        if verbose:
            print(f"Loss of epoch {epoch}: {epoch_loss}")

    return model