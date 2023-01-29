import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import random_split
from feature import encoder,Res_Assort_Net

np.random.seed(1)
num_p = 11
Z = np.random.normal(0,1,size=(num_p-1,8))*4#商品特征
Z = np.vstack((Z,np.zeros(8)))#11*8
cus = np.random.normal(0,1,size=(4,6))*4#顾客特征
np.save(r'products10.npy',Z)
np.save(r'cus4.npy',cus)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#实例化网络，建立训练模型
torch.manual_seed(0)
product_encoder = encoder(8,2,40,20).to(device)
cus_encoder = encoder(6,2,40,20).to(device)
net = Res_Assort_Net(22, 1, 22, 2).to(device)
print('cuda:',torch.cuda.is_available())

def ass(prod_num=[0,1,2,3,100],cus_type=0):
    prod = torch.from_numpy(Z[prod_num]).float().reshape((1,5,-1))
    cus_ = torch.from_numpy(np.repeat(cus[cus_type].reshape((1,1,6)),
                                      5,1)).float()
    return prod,cus_
ass_onehot = torch.tensor([[1,1,1,1,1]]).float()

'''for t in range(4):
    prod,cus_ = ass(prod_num=[25,0,23,77,100],cus_type=t)
    with torch.no_grad():
        e_prod = product_encoder(prod)
        e_cust = cus_encoder(cus_)
        latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*35
        y_hat = net(latent_uti, ass_onehot)
        print(torch.softmax(y_hat,1))'''

torch.save(product_encoder, r'resnet/product_encoder_simul.pth')
torch.save(cus_encoder, r'resnet/cus_encoder_simul.pth')
torch.save(net, r'resnet/net_simul.pth')