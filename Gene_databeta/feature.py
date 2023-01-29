import torch
torch.manual_seed(0)
import torch.nn as nn
import  torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
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

class simulator(nn.Module):
    def __init__(self, product_encoder, cus_encoder, Res_Assort_Net):
        super().__init__()
        self.product_encoder = product_encoder
        self.cus_encoder = cus_encoder
        self.Res_Assort_Net = Res_Assort_Net
    def forward(self,feature,ass_onehot):#应该输入tensor
        prod = feature[:, :, :8]
        cus = feature[:, :, 8:]
        e_prod = self.product_encoder(prod)
        e_cust = self.cus_encoder(cus)
        latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*35
        y_hat = self.Res_Assort_Net(latent_uti, ass_onehot)
        prob = nn.Softmax(1)(y_hat)
        return prob#选择各个元素的概率


def train_test_split(data):
    random.seed(0)
    srch_id_list = list(data["srch_id"].unique())
    random.shuffle(srch_id_list)
    train_data = data[data["srch_id"].isin(srch_id_list[0: round(len(srch_id_list)*0.8)])]
    test_data = data[data["srch_id"].isin(srch_id_list[round(len(srch_id_list)*0.8): ])]
    return train_data, test_data
import cvxpy as cp
def estimate_MNL_beta(train_data):
    train_data_srch_id=list(train_data["srch_id"].unique())
    #fit beta on data
    beta=cp.Variable(10)
    LL=cp.Constant(0)
    for id_ in train_data_srch_id:#对每一个search id
        query_data = train_data[train_data["srch_id"]==id_]#这一次seach被推荐的所有酒店数据
        id_list_all = query_data["prop_id"]#这一次seach被推荐的所有酒店id
        if query_data["booking_bool"].sum()==1:
            #purchase prob
            temp1=[0]
            temp2=[0]
            temp1+=[beta@query_data[query_data["booking_bool"]==1].values[0][3:]]
            for i in id_list_all:
                temp2+=[beta@query_data[query_data["prop_id"]==i].values[0][3:]]
            LL+=cp.sum(cp.vstack(temp1))-cp.log_sum_exp(cp.vstack(temp2))
        else:
            #no purchase prob
            temp=[0]
            for i in id_list_all:
                temp+=[beta@query_data[query_data["prop_id"]==i].values[0][3:]]
            LL+=-cp.log_sum_exp(cp.vstack(temp))
    objective=cp.Maximize(LL)
    constraints=[]
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    return list(list(prob.solution.primal_vars.values())[0])
def MNL_out_of_sample_log_likelihood(test_data,beta):
    test_data_srch_id=list(test_data["srch_id"].unique())
    LL=0
    CE = []
    for id_ in test_data_srch_id:#对每一个search id
        query_data = test_data[test_data["srch_id"]==id_]
        id_list_all = query_data["prop_id"]
        if query_data["booking_bool"].sum()==1:
            #purchase prob
            temp2 = 1
            temp1 = beta@query_data[query_data["booking_bool"]==1].values[0][3:]
            for i in id_list_all:
                temp2 += np.exp(beta@query_data[query_data["prop_id"]==i].values[0][3:])
            CE.append(-temp1 + np.log(temp2))
            LL += temp1 - np.log(temp2)
        else:
            #no purchase prob
            temp=0
            for i in id_list_all:
                temp += np.exp(beta@query_data[query_data["prop_id"]==i].values[0][3:])
            CE.append(np.log(temp))
            LL += -np.log(temp)
    return LL,np.mean(CE)
