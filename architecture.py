import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import time

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)        
        m.bias.data.fill_(0.01)

class Autoencoder(nn.Module):

    # LAYER_SIZES = [14, 9, 3]
    LAYER_SIZES = [24, 10, 6, 3]
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):

        self.emb_artists = nn.Embedding(27621, 5)
        self.emb_genres = nn.Embedding(2663, 5)

        # encoding architecture
        self.enc1 = nn.Linear(
            in_features = self.LAYER_SIZES[0], 
            out_features = self.LAYER_SIZES[1])
        self.enc2 = nn.Linear(
            in_features = self.LAYER_SIZES[1], 
            out_features = self.LAYER_SIZES[2])
        self.enc3 = nn.Linear(
            in_features = self.LAYER_SIZES[2], 
            out_features = self.LAYER_SIZES[3])

        # decoding architecture
        self.dec1 = nn.Linear(
            in_features = self.LAYER_SIZES[3], 
            out_features = self.LAYER_SIZES[2])
        self.dec2 = nn.Linear(
            in_features = self.LAYER_SIZES[2], 
            out_features = self.LAYER_SIZES[1])
        self.dec3 = nn.Linear(
            in_features = self.LAYER_SIZES[1], 
            out_features = self.LAYER_SIZES[0])

    def transform(self, x):
        print("TRANSFORMERS")
        # print(x[0].long())
        # prev = time.time()
        index_artists = [np.where(r==1) for r in x[0].numpy()]
        print("got index: ", index_artists)
        index_genre = [np.where(r==1) for r in x[2].numpy()]
        artists_embedding = self.emb_artists(index_artists)
        genres_embedding = self.emb_genres(index_genre)
        flattened_genres = genres_embedding[:,0,0,:]
        flattened_artist = artists_embedding[:,0,0,:]
        # print(flattened_artist.shape)
        # print(flattened_genres.shape)
        x = torch.cat((flattened_artist, x[1], flattened_genres), 1)
        # print("needed: ", (time.time() - prev))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self, x):

        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        # x = F.relu(self.enc3(x))
        
        return x
    
    def decode(self, x):
        # x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))

        return x



