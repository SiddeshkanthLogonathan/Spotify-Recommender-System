from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from visualization import DataVisualizer
import pandas as pd


default_model = Autoencoder()
default_num_epochs = 2
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

    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        prev = time.time()
        for index, batch in enumerate(dataloader):
            if index % 200 == 0:
                print("patch learned! ", index, " of ", len(dataloader))
            optimizer.zero_grad()
            # train_data = pad_sequence(batch, batch_first=True).double()
            outputs = model(batch)
            loss = criterion(outputs, batch.training_label)
            if(index % 100 == 0):
                print("current loss: ", loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        print("lr:", scheduler.get_last_lr())
        print(running_loss)
        loss = running_loss / len(dataloader)
        # animator.add(epoch,(loss))
        train_loss.append(loss)

        print('Epoch {} of {}, Train Loss: {:.3f}, Time spent: {}'.format(
            epoch + 1, num_epochs, loss, time.time()-prev))

        #print(train_loss)
    return train_loss, model

def main():
    train_loss, model = train(num_epochs=2)

    dl = SpotifyRecommenderDataLoader(default_dataset, batch_size=1000)
    batch = next(iter(dl))
    with torch.no_grad():
        encodings = model.encode(batch.training_label).numpy()

    print(encodings)

    df = pd.DataFrame.from_dict({'x': encodings[:, 0], 'y': encodings[:, 1], 'z': encodings[:, 2]})

    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
    )])
    fig.show()


if __name__ == "__main__":
    main()