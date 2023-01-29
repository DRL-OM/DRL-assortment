import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import random_split
from feature import encoder,Res_Assort_Net,simulator

np.random.seed(1)
num_p = 11
num_pf = 8
Z = np.random.normal(0,2,size=(num_p-1,num_pf))*4#商品特征
Z = np.vstack((Z,np.zeros(num_pf)))#11*8

num_cf = 4
num_cus_type = 4
cus = np.random.normal(0,2,size=(num_cus_type,num_cf))*4#顾客特征

Z = np.load('simul/p10_random.npy')
cus = np.load('simul/c4_random.npy')

prod_dup0 = np.repeat(Z.reshape(1, -1), num_cus_type, axis=0).reshape(num_p*num_cus_type, -1)
cus_dup0 = np.repeat(cus,num_p,axis=0)
concat_feature0 = np.concatenate((prod_dup0,cus_dup0),axis=1).reshape(num_cus_type,num_p,-1)

#np.save(r'p100c4/p'+str(num_p-1)+'_random.npy',Z)
#np.save(r'p100c4/c4_random.npy',cus)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

#实例化网络，建立训练模型
torch.manual_seed(0)
product_encoder = encoder(8,2,40,20)
cus_encoder = encoder(4,2,40,20)
net = Res_Assort_Net(2*num_p, 1, 2*num_p, 2)

product_encoder = torch.load(r'simul/Rproduct_encoder_simul.pth')#,map_location='cpu'
cus_encoder = torch.load(r'simul/Rcus_encoder_simul.pth')
Res_Assort_Net_ =  torch.load(r'simul/Rnet_simul.pth')
Resnet = simulator(product_encoder,cus_encoder,Res_Assort_Net_).to(device)
print('cuda:',torch.cuda.is_available())


'''
l_ = np.array([1,1,1,1]+[0]*96)
assortment = np.repeat(l_[np.newaxis,:],num_cus_type,0)
arriving_seg = np.array([[0],[1],[2],[3]])
batch = 4

multiplier = (np.hstack((assortment, np.ones((batch, 1))))).reshape(batch, num_p, 1)
multiplier = np.concatenate((np.repeat(multiplier, num_pf, axis=2), np.ones((batch, num_p, num_cf))), axis=2)
arriving_seg_feature = concat_feature0[arriving_seg.ravel(), :]
input_1 = torch.from_numpy(multiplier * arriving_seg_feature).float().to(device)  #product feature加cus feature
ass_onehot = torch.from_numpy(np.hstack((assortment, np.ones((batch, 1))))).float().to(device)
# 关键语句
with torch.no_grad():
    prob = Resnet(input_1, ass_onehot) 
print(prob)



torch.save(product_encoder, 'p100c4/R100product_encoder_simul.pth')
torch.save(cus_encoder, 'p100c4/R100cus_encoder_simul.pth')
torch.save(net, 'p100c4/R100net_simul.pth')'''


card = 4
import pandas as pd
import random
#读取simulator数据
ResNet = simulator(product_encoder, cus_encoder, net).to(device)
print('start sampling data for MNL!')
def generate(srch_num=5000):
    # 生成ass，得到choice
    random.seed(0)
    srch_id = np.arange(srch_num)
    srch_id = np.repeat(srch_id, card).reshape(srch_num * card, 1)
    ass_list = []
    cus_list = []
    for i in range(srch_num):
        ass_list.append(random.sample(list(range(num_p-1)), card))
        cus_list.append(random.sample(list(range(num_cus_type)), 1))
    ass_list0 = np.hstack((np.array(ass_list), np.zeros((srch_num, 1), dtype=int) + num_p-1))  # 每一次都包括了不选
    # 放进最后的表格的，不包括不选
    p_feature = Z[np.array(ass_list).ravel()].reshape((srch_num, card, -1))
    # resnet要所有十一个特征，没放的特征是0
    multi = np.zeros((srch_num, num_p, num_pf))
    multi[np.repeat(np.arange(srch_num), card+1), ass_list0.ravel(), :] = 1  # 放了的地方是1，当做乘子
    p_feature0 = np.repeat(Z.reshape(1, num_p, num_pf), srch_num, 0) * multi  # (5000, 11, 8)
    c_fea = cus[np.array(cus_list)]
    c_feature = np.repeat(c_fea, card, 1)  # 放进最后的表格的
    c_feature0 = np.repeat(c_fea, num_p, 1)  # resnet
    feature = np.concatenate((p_feature, c_feature), 2)  # 放进最后的表格的
    feature0 = np.concatenate((p_feature0, c_feature0), 2)  # resnet
    ass_onehot = np.zeros((srch_num, num_p-1))
    ass_onehot[np.repeat(np.arange(srch_num), card), np.array(ass_list).ravel()] = 1
    ass_onehot = np.hstack((ass_onehot, np.ones((srch_num, 1))))
    fff = torch.from_numpy(feature0).to(torch.float32)
    aaa = torch.from_numpy(ass_onehot).to(torch.float32)
    purchase_prob = ResNet(fff.to(device), aaa.to(device)).cpu().detach().numpy()
    impression_data = np.zeros((1, num_pf+num_cf+1))
    for i in range(srch_num):
        p_ = purchase_prob[i]
        index_ = np.random.choice(list(range(num_p)), p=p_.ravel())  # 十个里面的哪一个
        # ass_list[index]
        choice = np.zeros((card, 1))
        if index_ != (num_p-1):
            choice[ass_list[i].index(index_)] = 1  # ass_list[i]包括的是商品代号，要知道的是选了这个列表里面的第几个
        impression_data = np.vstack((impression_data,
                                     np.hstack((feature[i], choice))
                                     ))
    impression_data = np.hstack((srch_id, impression_data[1:, :]))
    return pd.DataFrame(impression_data)

import os
import scipy.io as scio
for num in [100]:#0,5000,10000,20000,30000]:
    table = generate(num)
    MNL_data = table.copy()
    dat_ = np.array(MNL_data)
    Ind = list(MNL_data.index)
    Col = list(MNL_data.columns)
    scio.savemat(os.getcwd() + '/Mat/' + 'Rgene_data'+str(num)+'.mat',{'data':dat_,'index':Ind,'cols':Col})
