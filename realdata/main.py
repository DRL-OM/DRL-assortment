from feature import encoder,Res_Assort_Net,simulator,Gate_Assort_Net
from uti import setup_logger,generate_set
from net import A2C
import torch
import numpy as np
from numpy import *
import pandas as pd
from arg import init_parser
import os
import scipy.io as io
import matplotlib.pyplot as plt
import time
import json
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
############################
### groundtruth model
prop_features = np.load(args.p_file)
product_encoder = torch.load(r'resnet/ex_product_encoder.pth')
cus_encoder = torch.load(r'resnet/ex_cus_encoder.pth')
Res_Assort_Net_ = torch.load(r'resnet/ex_net.pth')
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_).to(device)
products_price = prop_features[:-1,5]*72.28+155.8
logger.info("products price: {}".format(products_price))
products_price = products_price/np.max(products_price)
with open('../../expedia/new/seqdata.json', 'r') as f:
    args.seqdata = json.loads(f.read())
############################
### 设定实验参数
initial_inventory = np.array([args.ini_inv]*args.num_products)
T = 120
logger.info("Load Factor {}".format(T/np.sum(initial_inventory)))
############################
### 划分训练和测试
train_sequences = list(args.seqdata.values())[:160]
test_sequences = list(args.seqdata.values())[160:]
############################
from train import train,test,plot_box
if args.test:
    #读取MNL参数
    MNL_para1 = io.loadmat('MNL/20round_beta1.mat')
    MNL_para1 = MNL_para1['var'].reshape((-1,6))
    MNL_para2 = io.loadmat('MNL/20round_beta2.mat')
    MNL_para2 = np.vstack((  MNL_para1 , MNL_para2['var'].reshape((-1,6)) ))
    MNL_para3 = io.loadmat('MNL/20round_beta3.mat')
    MNL_para3 = np.vstack((  MNL_para2 , MNL_para3['var'].reshape((-1,6)) ))
    MNL_para4 = io.loadmat('MNL/20round_beta4.mat')
    MNL_para = np.vstack((  MNL_para3 , MNL_para4['var'].reshape((-1,6)) ))
    
    test(test_sequences,ResNet,MNL_para,
            initial_inventory,products_price,args,logger,load=True,plot= False)
else:
    os.mkdir('MNL/'+name)
    train(args,ResNet,products_price,initial_inventory,train_sequences,logger)
    
