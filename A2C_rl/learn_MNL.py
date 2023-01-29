import numpy as np
import torch
from Resnet import encoder,Res_Assort_Net,simulator
import pandas as pd
#读取simulator数据
product_encoder = torch.load('product_encoder2.pth')
cus_encoder = torch.load('cus_encoder2.pth')
Res_Assort_Net_ =  torch.load('net2.pth')
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def generate():
    srch_id = np.arange(2000)
    srch_id = np.repeat(srch_id,35).reshape(2000*35,1)
    # 特征
    feature1 = np.random.randint(2,size = (2000*35,3))
    feature2 = (np.random.random((2000*35, 5)) - 0.5) * 2 #-1到1
    feature3 = (np.random.random((2000, 5)) - 0.5) * 2  # -1到1
    feature3 = np.repeat(feature3,35,axis=0)
    feature4 = np.random.randint(2, size=(2000, 1))
    feature4 = np.repeat(feature4, 35, axis=0)
    cus = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    seg_prob = [0.4, 0.3, 0.1, 0.2]
    input_sequence = np.random.choice(a=np.arange(4), size=2000, replace=True, p=seg_prob)
    feature5 = cus[input_sequence]
    feature5 = np.repeat(feature5, 35, axis=0)

    feature = np.hstack((feature1, feature2))
    for i in range(2000):
        feature[i*35] = 0
    feature = np.hstack((feature, feature3))
    feature = np.hstack((feature, feature4))
    feature = np.hstack((feature, feature5))
    ass = np.ones((2000 * 35, 35))
    feature = np.hstack((feature, ass))
    feature = feature.reshape(2000,35,-1)

    feature = torch.from_numpy(feature).float().to(device)
    purchase_prob = ResNet(feature).cpu().detach().numpy()
    book_bool = np.array([[0]], dtype=np.int)
    for i in range(2000):
        purchase = np.zeros((35, 1), dtype=np.int)
        p_ = purchase_prob[i]
        index = np.random.choice(list(range(len(p_))), p=p_.ravel())
        purchase[index] = 1
        book_bool = np.vstack((book_bool, purchase))
    book_bool = book_bool[1:]

    feature = feature.cpu().numpy()
    impression_data = np.hstack((srch_id, srch_id))#第二列没用
    impression_data = np.hstack((impression_data, book_bool))
    impression_data = np.hstack((impression_data, feature.reshape(2000*35,-1)[:,:18]))
    return pd.DataFrame(impression_data, columns=['srch_id', 'dummy', 'booking_bool', 'prop_brand_bool',
                                                   'promotion_flag', 'random_bool', 'prop_starrating', 'prop_review_score',
                                                   'prop_location_score1', 'prop_log_historical_price', 'price_per_night',
                                                   'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count',
                                                   'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool',
                                                   'cus_country1', 'cus_country2', 'cus_country3', 'cus_country4'])

table = generate()

import os
import scipy.io as scio
MNL_data = table.copy()
#MNL_data.drop(columns=['prop_id'])
MNL_data = MNL_data.reset_index(drop=True)
dat_ = np.array(MNL_data)
Ind = list(MNL_data.index)
Col = list(MNL_data.columns)
scio.savemat(os.getcwd() + '/Mat/' + 'gene_data.mat',{'data':dat_,'index':Ind,'cols':Col})



