from data_loading import SpotifyRecommenderDataset
from architecture import Autoencoder
import torch



def train_autoencoder(model: Autoencoder, dataloader, verbose=True, num_epochs=10):
    criterion = torch.nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            labels = batch
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(dataloader)
        
        if verbose:
            print(f"Loss of epoch {epoch + 1}: {epoch_loss}")

    return model