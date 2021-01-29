import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_size):
         super().__init__()
         self.l1 = nn.Linear(input_size, input_size * 3)
         self.l2 = nn.Linear(input_size * 3, input_size * 2)
         self.l3 = nn.Linear(input_size * 2, 1)


    def forward(self, inputs):
        o1 = F.relu(self.l1(inputs))
        o2 = F.relu(self.l2(o1))
        o3 = self.l3(o2)
        return o3
