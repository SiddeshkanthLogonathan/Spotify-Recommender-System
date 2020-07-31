import torch
import torch.nn as nn
from architecture import Autoencoder


def test(model: Autoencoder, input_tensor: torch.tensor) -> int:
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        outputs = model(input_tensor)
        return criterion(outputs, input_tensor)
