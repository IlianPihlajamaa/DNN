import sys
import torch
from numpy import log
from torch import nn



class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__()
        self.n_in = kwargs['n_in']
        self.n_out = kwargs['n_out']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.bias = kwargs['bias']
        self.device = kwargs['device']

        if(len(self.net_width)!=(self.net_depth-2)):
            print('Error: the number of layers and their size does not match')
            sys.exit()


        layers = []
        layers.append(
            nn.Linear(
                self.n_in,
                self.n_out if self.net_depth == 1 else int(self.net_width[0]),
                bias=self.bias)
            )
        layers.append(
                nn.GELU())
        for count in range(self.net_depth - 3):
            layers.append(
                nn.Linear(
                    int(self.net_width[count]),
                    int(self.net_width[count+1]), 
                    bias=self.bias                    )
                )
            layers.append(
                nn.GELU())
        if self.net_depth >= 2:
            layers.append(
                nn.Linear(
                    int(self.net_width[-1]),
                    self.n_out,
                    bias=self.bias)
                )
         

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x_out = self.net(x)
        return x_out
