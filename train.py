# Age Prediction from Voice Input Project

import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    #def forward(self, x):