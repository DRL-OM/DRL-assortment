from feature import Gate_Assort_Net
from uti import setup_logger,generate_set
from net import A2C
import torch
import numpy as np
from numpy import *
from arg import init_parser
from train import train,test,plot_box
import os
import scipy.io as io
import matplotlib.pyplot as plt
import time
import json

###  设置args和logger
parser = init_parser('A2C')
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
products_price = [1.838419913,1.992458678,1.874724518,1.515151515,3.28728191,2.695362718,1.032467532,4.485454545,1.57983683,1.02]
args.rank_list = np.load('GT/ranked_lists_use.npy')
args.cus_type = np.load('GT/cus_types.npy')
seg_prob = np.load('GT/cusseg_prob.npy')
with open('GT/seqdata.json', 'r') as f:
    args.seqdata = json.loads(f.read())
with open('GT/transdata.json', 'r') as f:
    args.transdata = json.loads(f.read())
############################
### 加载fitted model
Gated_net = []
for i in range(4):
    Gated_net.append(torch.load('GT/ResNet'+str(i)+'.pth'))
MNL_para = np.load('GT/Learned_MNL_weight.npy')    
############################
###  设定实验参数
initial_inventory = np.array([args.ini_inv]*10)
logger.info("Load Factor {}".format(100/np.sum(initial_inventory)))
############################


#划分训练和测试
train_sequences = list(args.seqdata.values())[:160]
test_sequences = list(args.seqdata.values())[160:]
from train import train,test,plot_box
if args.detail:
    test(test_sequences,MNL_para,
            initial_inventory,products_price,args,logger,load=True,plot= False)
elif args.only_test:
    test(test_sequences,MNL_para,
            initial_inventory,products_price,args,logger,load=True,plot= False)
    logger.info(name+"completed")
else:
    train(args,Gated_net,products_price,initial_inventory,train_sequences,logger)
    #test(test_sequences,MNL_para,initial_inventory,products_price,args,logger,load=True,plot= False)
    logger.info(name+"completed")

#A2C2023-02-15-21-07-45 
