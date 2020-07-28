from data_loading import SpotifyRecommenderDataset, SpotifyRecommenderDataLoader
from architecture import Autoencoder
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time


default_model = Autoencoder()
default_num_epochs = 2
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
            train_data = batch.training_label
            if(index % 100 == 0):
                print("patch learned! ", index, " of ", len(dataloader))
            optimizer.zero_grad()
            # train_data = pad_sequence(batch, batch_first=True).double()
            outputs = model(batch)
            loss = criterion(outputs, train_data)
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

        print(train_loss)
    return train_loss

plt.plot(train()[2:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

torch.save(default_model, './model.pth')
