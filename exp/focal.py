import torch
import torch.nn as nn
import numpy as np

class LearnFocal(nn.Module):
    def __init__(self, init_focal=None):
        super(LearnFocal, self).__init__()

        if init_focal is None:
            self.focal = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        else:
            self.init_focal = nn.Parameter(init_focal, requires_grad=False)
            self.focal = nn.Parameter(self.init_focal, requires_grad=True)
    
    def forward(self):
        return self.focal