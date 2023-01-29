import numpy as np
import torch
from feature import encoder,Res_Assort_Net,simulator
import pandas as pd
import random
#读取simulator数据
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
product_encoder = torch.load(r'../Gene_databeta/ex_product_encoder.pth')
cus_encoder = torch.load(r'../Gene_databeta/ex_cus_encoder.pth')
Res_Assort_Net_ =  torch.load(r'../Gene_databeta/ex_net.pth')
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_).to(device)

num_p = 58#有一个是不买
num_cus_type = 4
seg_prob = [0.4,0.3,0.1,0.2]
num_pf = 6
num_cf = 4

card = 20
Z = np.load('../Gene_databeta/save/expedia_prop_features.npy')#商品特征
cus = np.eye(4)
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
        if index_ != num_p-1:
            choice[ass_list[i].index(index_)] = 1  # ass_list[i]包括的是商品代号，要知道的是选了这个列表里面的第几个
        impression_data = np.vstack((impression_data,
                                     np.hstack((feature[i], choice))
                                     ))
    impression_data = np.hstack((srch_id, impression_data[1:, :]))
    return pd.DataFrame(impression_data)
table = generate(20000)

import os
import scipy.io as scio
MNL_data = table.copy()
dat_ = np.array(MNL_data)
Ind = list(MNL_data.index)
Col = list(MNL_data.columns)
scio.savemat('resnet/gene_data20000.mat',{'data':dat_,'index':Ind,'cols':Col})



