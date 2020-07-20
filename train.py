from data_loading import SpotifyRecommenderDataset
from architecture import Autoencoder
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
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

decayRate = 0.5
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decayRate)

trainloader = torch.utils.data.DataLoader(d, shuffle=True, batch_size=256, collate_fn=own_collate_fn)

def train(n, num_epochs):

    #animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs], legend=['train loss'])
    train_loss = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for value in trainloader:
            my_optim.zero_grad()
            with torch.no_grad():
                value = n.transform(value)
            outputs = n(value)
            loss = criterion(outputs, value)
            loss.backward()
            my_optim.step()
            running_loss += loss.item()
        my_lr_scheduler.step()
        print("lr:",my_lr_scheduler.get_last_lr())
        print(running_loss)
        loss = running_loss / len(trainloader)
        #animator.add(epoch,(loss))
        train_loss.append(loss)

        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, num_epochs, loss))

        print(train_loss)
    return train_loss

plt.plot(train(n, 20)[2:])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print(n.enc1.weight)

counter = 0
for data in d:
    print("data: ", data)
    output = n.encode(data)
    print("encoded: ", output)
    print("decoded: ", n.decode(output))
    if(counter == 2):
        break
    counter += 1