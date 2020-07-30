from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder, GenreEmbedder
import torch



def train_autoencoder(model: Autoencoder, dataloader, verbose=True):
    num_epochs = 10
    criterion = torch.nn.MSELoss()
    learning_rate = 0.01
    # default_decay_rate = 0.9

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
    #                 amsgrad=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=default_decay_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.training_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)
        #scheduler.step()
        if verbose:
            print(f"Loss of epoch {epoch}: {epoch_loss}")

    return model

def train_genre_embedder(model: GenreEmbedder, dataloader, verbose=True):
    num_epochs = 10
    criterion = torch.nn.KLDivLoss()
    learning_rate = 0.05
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            input = batch[0]
            label = batch[1]
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)
        if verbose:
            print(f"Loss of epoch {epoch}: {epoch_loss}")

    return model