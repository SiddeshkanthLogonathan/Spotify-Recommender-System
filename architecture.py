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
        self.emb_genres = nn.Embedding(2664, 5)

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
        print("TRANSFORMEN MERS")
        out = None
        for i in range(x[0].shape[0]):
            index_artists = [np.where(r==1) for r in (x[0].squeeze()[i]).numpy()]
            index_artists = torch.tensor(index_artists[0][0])

            index_genre = [np.where(r==1) for r in x[2][i].numpy()]
            index_genre = torch.tensor(index_genre[0][0])
            if(len(index_genre) == 0):
                index_genre = torch.tensor([2663])

            artists_embedding = self.emb_artists(index_artists)
            genres_embedding = self.emb_genres(index_genre)

            artists_embedding = torch.sum(artists_embedding, dim = 0) / artists_embedding.shape[0]
            genres_embedding = torch.sum(genres_embedding, dim = 0) / genres_embedding.shape[0]
            temp_x = torch.cat((artists_embedding.double(), x[1][i].double(), genres_embedding.double()), 0)
            if out is None:
                out = temp_x.unsqueeze(0)
            else: 
                out = torch.cat((out, temp_x.unsqueeze(0)), dim = 0)
        return out

    def forward(self, x):
        x = self.encode(x.double())
        x = self.decode(x.double())

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



