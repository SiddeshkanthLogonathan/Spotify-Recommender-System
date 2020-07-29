from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
import torch
import torch.nn as nn
import time
import pandas as pd
import tqdm


default_model = Autoencoder()
default_num_epochs = 10
default_dataset = SpotifyRecommenderDataset()
default_batch_size = 256
default_dataloader = SpotifyRecommenderDataLoader(default_dataset, batch_size=default_batch_size)
default_criterion = nn.MSELoss()
default_learning_rate = 0.01
default_optimizer = torch.optim.Adam(params=default_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                     amsgrad=False)
default_decay_rate = 1.0
default_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=default_optimizer, gamma=default_decay_rate)


def train(model=default_model, num_epochs=default_num_epochs, dataloader=default_dataloader,
          optimizer=default_optimizer, criterion=default_criterion, scheduler=default_scheduler):

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
        print(f"Loss of epoch {epoch}: {epoch_loss}")

    return model

def main():
    model = train()

    dl = SpotifyRecommenderDataLoader(default_dataset, batch_size=len(default_dataset))
    batch = next(iter(dl))
    with torch.no_grad():
        encodings = model.encode(batch.training_label).numpy()

    default_dataset.add_encoding_columns(encodings)

if __name__ == "__main__":
    main()