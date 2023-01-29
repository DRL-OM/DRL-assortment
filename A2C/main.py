from feature import encoder,Res_Assort_Net,simulator
from uti import setup_logger,generate_set
from net import A2C
import torch
import numpy as np
from numpy import *
import pandas as pd
from arg import init_parser
from train import train,test,plot_box
import os
import scipy.io as io
import matplotlib.pyplot as plt
import time
parser = init_parser('Reinforce')
args = parser.parse_args()
device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
args.device = device
name = 'A2C' + time.strftime('%Y-%m-%d-%H-%M-%S')
args.name = name
stream = os.path.join('log', name)
logger = setup_logger(name='Train', level=20, stream = stream)
args.logger = logger
logger.info(args)
logger.info(stream)
'''
data = pd.read_csv('../../expedia/real_data.csv',index_col=0)
prop_features = []#每个hotel的特征
for prop in data['prop_id'].unique():
    prop_features.append(data[data['prop_id']==prop].iloc[0,3:9])
prop_features.append(np.zeros(6))
prop_features = np.array(prop_features)#(58, 6)shape
np.save('prop_features.npy',prop_features)'''
prop_features = np.load('prop_features.npy')
#读取MNL参数
if bool(args.no_cus):
    MNL_para = io.loadmat('resnet/gene_beta_P.mat')
else:
    MNL_para = io.loadmat('resnet/gene_beta.mat')
MNL_para = MNL_para['beta'].ravel()#(10,)
#读取simulator数据
product_encoder = torch.load(r'../Gene_databeta/ex_product_encoder.pth')
cus_encoder = torch.load(r'../Gene_databeta/ex_cus_encoder.pth')
Res_Assort_Net_ = torch.load(r'../Gene_databeta/ex_net.pth')
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_).to(device)
#设定实验参数
products_price = (prop_features[:-1,5]*261.9+151)/200#(57,)
logger.info("products price: {}".format(products_price*200))
seg_prob = [0.4,0.3,0.1,0.2]
seg_prob = [0.25,0.256,0.244,1-0.25-0.256-0.244]
initial_inventory = np.array([args.ini_inv]*args.num_products)
T = args.selling_length
logger.info("Load Factor {}".format(T/np.sum(initial_inventory)))

'''
import csv
from sklearn.model_selection import train_test_split
with open('sequences.csv','r',encoding='utf-8') as csvfile:#20000个sequence
    reader = csv.reader(csvfile)
    rows = [list(map(int, row[1:])) for row in reader][1:]
rows = rows[:args.total_episode]
train_set, test_set = \
    train_test_split(np.array(rows), test_size=args.test_size, random_state=args.seed)#19600,400
test_set, val_set = \
    train_test_split(test_set, test_size=args.val_size, random_state=args.seed)
train_set = np.split(train_set,args.train_batch_size)
T_list = []#每一个batch的长度
for i in range(int(19600/50)):
    T_now = np.random.randint(T-10,T+10)
    T_list.append(T_now)'''


if args.only_test:
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #读取sequence数据
        val_set,test_set = generate_set(args,T,seed_,seg_prob)
        #for epoch in range(args.epoch_num):
        OA_list,myopic_list,E_IB_list,seller_list = test(ResNet,MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
        
    plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
             E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))
else:
    val_set,test_set = generate_set(args,T,0,seg_prob)
    train(args,ResNet,seg_prob,products_price,initial_inventory,val_set,logger)
    
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #读取sequence数据
        val_set,test_set = generate_set(args,T,seed_,seg_prob)
        #for epoch in range(args.epoch_num):
        OA_list,myopic_list,E_IB_list,seller_list = test(ResNet,MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
        
    plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
             E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))


#A2C2022-12-05-17-03-36 A2C2022-12-05-17-28-17 不好 lr_min=0.0001太大了
#A2C2022-12-05-17-09-26 A2C2022-12-05-17-24-26 有波动，还可以 lr_decay_lambda=0.99 lr_min=1e-05


#A2C2022-12-05-20-12-50很好A2C2022-12-05-20-12-50.pdf

#A2C2022-12-10-11-09-09.pdf 很好

#A2C2022-12-17-16-23-40改进了sample

#A2C2022-12-17-18-36-15很好