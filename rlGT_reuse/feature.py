import torch
torch.manual_seed(0)
import torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class resBlock(nn.Module):
    def __init__(self,input_dim,depth,width):
        super().__init__()
        fully_conn = []
        if depth == 1:
            fully_conn.append(nn.Linear(input_dim,input_dim))
        else:
            for d in range(depth):
                fully_conn.append(nn.Linear(input_dim,width))
                fully_conn.append(nn.ReLU())
                fully_conn.append(nn.Linear(width, input_dim))
        self.fully_conn = nn.Sequential(*fully_conn)
        self.shortcut = nn.Sequential()
    def forward(self,x):
        return self.fully_conn(x) + self.shortcut(x)

class Res_Assort_Net(nn.Module):
    def __init__(self, input_dim, res_depth, res_width, num_blocks):
        super().__init__()
        res_blocks = []
        for i in range(num_blocks):
            res_blocks.append(resBlock(input_dim, res_depth, res_width))
        self.res_blocks = nn.Sequential(*res_blocks)
    def forward(self,x):#应该输入tensor
        score = self.res_blocks(x).mul(x)
        score[score==0]=-1e20
        return score#选择各个元素的概率


class Gate_Assort_Net(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, input_dim)
        )
    def forward(self,x):#应该输入tensor
        score = self.layers(x).mul(x)
        score[score==0]=-1e20
        return score#sotfmax之后是选择各个元素的概率


