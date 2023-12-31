import torch.nn as nn
import torch

class FCNN(nn.Module):

    """
        Define a Fully Connected (Feedforward) Neural Network.
    """
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        
        super().__init__()
        activation = nn.Tanh
        # activation = nn.ReLU
        # activation = nn.Sigmoid
        
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()
        ])
        
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()
            ]) for _ in range(N_LAYERS-1)
        ])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
    def forward(self, x):
        
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        
        return x
