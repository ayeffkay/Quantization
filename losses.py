import torch.nn as nn
import torch.nn.functional as F


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, targets):
        return F.mse_loss(outputs.view(-1), targets)
