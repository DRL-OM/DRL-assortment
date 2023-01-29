import torch
torch.manual_seed(0)
import torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class encoder(nn.Module):
    def __init__(self,input_dim,depth,width,output_dim):
        super().__init__()
        fully_conn = []
        if depth==1:
            fully_conn.append(nn.Linear(input_dim, output_dim))
        else:
            fully_conn.append(nn.Linear(input_dim, width))
            fully_conn.append(nn.ReLU())
            for d in range(depth-2):
                fully_conn.append(nn.Linear(width, width))
                fully_conn.append(nn.ReLU())
            fully_conn.append(nn.Linear(width, output_dim))
        self.fully_conn = nn.Sequential(*fully_conn)
    def forward(self,x):
        return self.fully_conn(x)

class resBlock(nn.Module):
    def __init__(self,input_dim,depth,width):
        super().__init__()
        fully_conn = []
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
        self.final_layer = nn.Linear(input_dim, int(input_dim/2))
    def forward(self,uti,ass):#应该输入tensor
        input_ = torch.cat((uti, ass), 1)#batch_size*70
        out = self.res_blocks(input_)
        score = self.final_layer(out).mul(ass)
        score[score==0]=-1e20
        return score#选择各个元素的概率



