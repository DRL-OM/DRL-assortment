import numpy as np
import random

num_p = 11
np.random.seed(0)

Z = np.load(r'prod10.npy')
cus = np.load(r'c4.npy')

import torch
from feature import encoder,Res_Assort_Net,simulator
import pandas as pd
#璇诲彇simulator鏁版嵁
product_encoder = torch.load(r'resnet/product_encoder_simul0.pth')#1对应MCCM产生的data
cus_encoder = torch.load(r'resnet/cus_encoder_simul0.pth')
Res_Assort_Net_ =  torch.load(r'resnet/net_simul0.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_).float().to(device)

def ass(prod_num=[0,1,2,3,10],cus_type=0):
    prod = torch.from_numpy(Z[prod_num]).float().reshape((1,5,-1))
    cus_ = torch.from_numpy(np.repeat(cus[cus_type].reshape((1,1,6)),
                                      5,1)).float()
    return prod,cus_

def generate(srch_num=5000):
    #生成ass，得到choice
    random.seed(0)
    srch_id = np.arange(srch_num)
    srch_id = np.repeat(srch_id,4).reshape(srch_num*4,1)
    ass_list = []
    cus_list = []
    for i in range(srch_num):
        ass_list.append(random.sample(list(range(10)), 4))
        cus_list.append(random.sample(list(range(4)), 1))
    ass_list0 = np.hstack((np.array(ass_list),np.zeros((srch_num,1),dtype=int)+10))#每一次都包括了不选
    #放进最后的表格的，不包括不选
    p_feature = Z[np.array(ass_list).ravel()].reshape((srch_num,4,-1))
    #resnet要所有十一个特征，没放的特征是0
    multi = np.zeros((srch_num,11,8))
    multi[np.repeat(np.arange(srch_num),5),ass_list0.ravel(),:] = 1#放了的地方是1，当做乘子
    p_feature0 = np.repeat(Z.reshape(1,11,8),srch_num,0)*multi  # (5000, 11, 8)
    c_fea = cus[np.array(cus_list)]
    c_feature = np.repeat(c_fea, 4, 1)#放进最后的表格的
    c_feature0 = np.repeat(c_fea, 11, 1)#resnet
    feature = np.concatenate((p_feature,c_feature),2)#放进最后的表格的
    feature0 = np.concatenate((p_feature0,c_feature0),2)#resnet
    ass_onehot = np.zeros((srch_num,10))
    ass_onehot[np.repeat(np.arange(srch_num),4),np.array(ass_list).ravel()] = 1
    ass_onehot = np.hstack((ass_onehot,np.ones((srch_num,1))))
    fff = torch.from_numpy(feature0).to(torch.float32)
    aaa = torch.from_numpy(ass_onehot).to(torch.float32)
    purchase_prob = ResNet(fff.to(device),aaa.to(device)).cpu().detach().numpy()
    impression_data = np.zeros((1,15))
    for i in range(srch_num):
        p_ = purchase_prob[i]
        index_ = np.random.choice(list(range(11)), p=p_.ravel())#十个里面的哪一个
        #ass_list[index]
        choice = np.zeros((4,1))
        if index_!=10:
            choice[ass_list[i].index(index_)] = 1#ass_list[i]包括的是商品代号，要知道的是选了这个列表里面的第几个
        impression_data = np.vstack((impression_data,
                                     np.hstack((feature[i], choice))
                                     ))
    impression_data = np.hstack((srch_id,impression_data[1:,:]))
    return pd.DataFrame(impression_data)


table = generate(20000)
import os
import scipy.io as scio
MNL_data = table.copy()
dat_ = np.array(MNL_data)
Ind = list(MNL_data.index)
Col = list(MNL_data.columns)
scio.savemat(os.getcwd() + '/' + 'gene_Rdata_MCCM20000.mat',
             {'data':dat_,'index':Ind,'cols':Col})



