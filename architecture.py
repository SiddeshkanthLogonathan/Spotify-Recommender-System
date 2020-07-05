import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.01)        
        m.bias.data.fill_(0.01)

class Autoencoder(nn.Module):

    LAYER_SIZES = [14, 10, 6, 3]
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.buildArchitecture()
        self.apply(init_weights)

    def buildArchitecture(self):
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

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        
        return x
    
    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.sigmoid(self.dec3(x))

        return x



