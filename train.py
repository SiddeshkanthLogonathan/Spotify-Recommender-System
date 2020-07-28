from data_loading import SpotifyRecommenderDataset
from architecture import Autoencoder
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# import d2l
import time
import matplotlib.pyplot as plt


n = Autoencoder()
n.buildArchitecture()

n = n.double()
d = SpotifyRecommenderDataset()

def own_collate_fn(batch):
    return batch

criterion = nn.MSELoss()
#optimizer = optim.Adam(n.parameters(), lr=0.0003)
my_optim = torch.optim.Adam(params=n.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

decayRate = 0.7
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decayRate)

trainloader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=256, collate_fn=own_collate_fn, num_workers=8)

def train(n, num_epochs):

    #animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss'])
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        prev = time.time()
        for index, value in enumerate(trainloader):
            if(index % 100 == 0):
                print("patch learned! ", index, " of ", len(trainloader))
            # prev = time.time()
            my_optim.zero_grad()
            # value = n.transform(value)
            value = pad_sequence(value, batch_first=True).double()
            # print(value)
            outputs = n(value)
            loss = criterion(outputs, value)
            if(index % 100 == 0):
                print("current loss: ", loss)
            loss.backward()
            my_optim.step()
            running_loss += loss.item()
        my_lr_scheduler.step()
        loss = running_loss / len(trainloader)
        print("lr:",my_lr_scheduler.get_last_lr())
        # print(running_loss)
        #animator.add(epoch,(loss))
        train_loss.append(loss)

        print('Epoch {} of {}, Train Loss: {:.3f}, Time spent: {}'.format(
            epoch + 1, num_epochs, loss, time.time()-prev))

        print(train_loss)
    return train_loss

plt.plot(train(n, 10)[2:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# print(n.enc1.weight)

torch.save(n, './model.pth')