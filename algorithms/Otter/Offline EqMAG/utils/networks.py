import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import Linear as lin
import math


class MLP(Module):
    def __init__(self, input_dim, output_dim,hidden_dim,num_hidden):
        super().__init__()

        assert num_hidden >= 1, "For MLP, num_hidden must be at least 1. This class does not support 0 hidden layers."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        
        layers = []
        
        first_layer = lin(self.input_dim,self.hidden_dim)
        layers.append(first_layer)
        layers.append(nn.Mish())
        
        for _ in range(0,self.num_hidden-1):
            body = lin(self.hidden_dim,self.hidden_dim)
            layers.append(body)
            layers.append(nn.Mish())
        
        head = lin(self.hidden_dim,self.output_dim)
        layers.append(head)
        self.model = nn.Sequential(*layers)

    
    def forward(self,x):
        o = self.model(x)
        return o
        


class ConditionalMLP(Module):
    def __init__(self, input_dim, output_dim,hidden_dim,num_hidden):
        super().__init__()

        assert num_hidden >= 1, "For ConditionalMLP, num_hidden must be at least 1. This class does not support 0 hidden layers."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        
        layers = []
        
        first_layer = lin(self.input_dim,self.hidden_dim)
        layers.append(first_layer)
        layers.append(nn.Mish())
        
        for _ in range(0,self.num_hidden-1):
            body = lin(self.hidden_dim,self.hidden_dim)
            layers.append(body)
            layers.append(nn.Mish())
        
        head = lin(self.hidden_dim,self.output_dim)
        layers.append(head)
        self.model = nn.Sequential(*layers)

    
    def forward(self,action,state):
        x = torch.concat([action,state],dim=-1)
        o = self.model(x)
        return o
        
    
        
        
        