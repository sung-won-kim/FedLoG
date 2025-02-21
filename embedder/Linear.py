import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,input,w=None,b=None):
        if w!=None:
            return torch.mm(input,w)+b
        else:
            return torch.mm(input,self.weight)+self.bias