import torch
from architecture import Autoencoder

net = torch.load('./model.pth')
net.eval()