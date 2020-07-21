from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
import torch
import torch.nn as nn
import time


default_model = Autoencoder()
default_num_epochs = 10
default_dataset = SpotifyRecommenderDataset()
default_batch_size = 256
default_dataloader = SpotifyRecommenderDataLoader(default_dataset, batch_size=default_batch_size, shuffle=True)
default_criterion = nn.MSELoss()
default_learning_rate = 0.01
default_optimizer = torch.optim.Adam(params=default_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                     amsgrad=False)
default_decay_rate = 0.5
default_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=default_optimizer, gamma=default_decay_rate)


def train(model=default_model, num_epochs=default_num_epochs, dataloader=default_dataloader,
          optimizer=default_optimizer, criterion=default_criterion, scheduler=default_scheduler):

    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        prev = time.time()
        for index, batch in enumerate(dataloader):
            label = batch.training_label
            print("patch learned! ", index, " of ", len(dataloader), " time spent: ", (time.time() - prev))
            prev = time.time()
            optimizer.zero_grad()
            #with torch.no_grad():
            outputs = model(batch)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print("lr:", scheduler.get_last_lr())
        print(running_loss)
        loss = running_loss / len(dataloader)
        # animator.add(epoch,(loss))
        train_loss.append(loss)

        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, num_epochs, loss))

        print(train_loss)
    return train_loss

train()