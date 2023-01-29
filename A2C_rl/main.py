from feature import encoder,Res_Assort_Net,simulator
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
#设置args和logger
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
#读取MNL参数
MNL_para = np.load('MNLbeta.npy')#长度为40的array，代表4类顾客对10个商品的utility
MNL_para = np.exp(MNL_para)
#读取simulator数据
cus_type = np.load('cus_type.npy')
rank_list = np.load('rank_list.npy')
args.cus_type = cus_type
args.rank_list = rank_list
#设定实验参数
#products_price = np.linspace(120,30,10)/100
products_price = np.linspace(2000,100,10)/2000#由大到小排列 120,20,-10
logger.info("products price: {}".format(products_price*2000))
seg_prob = [0.4,0.3,0.1,0.2]
#initial_inventory = np.random.randint(10, 30, size=args.num_products)
initial_inventory = np.array([args.ini_inv]*10)
T = args.selling_length
logger.info("Load Factor {}".format(T/np.sum(initial_inventory)))


if args.detail:
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    seed_ = 1
    val_set,test_set = generate_set(args,T,seed_,seg_prob)
    test_set = np.repeat(test_set,args.batch_size,0)
    OA_list,myopic_list,E_IB_list,seller_list = test(MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)

elif args.only_test:
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #读取sequence数据
        #seed_ = 1
        val_set,test_set = generate_set(args,T,seed_,seg_prob)
        if args.detail:
            test_set = np.repeat(test_set,args.batch_size,0)
        #for epoch in range(args.epoch_num):
        OA_list,myopic_list,E_IB_list,seller_list = test(MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
        
    plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
             E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))
else:
    val_set,test_set = generate_set(args,T,0,seg_prob)
    train(args,seg_prob,products_price,initial_inventory,val_set,logger)
    
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #读取sequence数据
        val_set,test_set = generate_set(args,T,seed_,seg_prob)
        #for epoch in range(args.epoch_num):
        OA_list,myopic_list,E_IB_list,seller_list = test(MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
        
    plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
             E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))


#A2C2022-12-02-12-00-09 表现不错 0.9的loadfactor只卖出了不到一半的东西
#A2C2022-12-02-14-07-17 lf1.4不错
#A2C2022-12-02-14-08-29 lf1.2没学好

#学习过程是有的，要比较是不是比myopic好
#A2C2022-12-03-15-48-48.pdf lf1.4的结果 三种方法差不多

#是不是lf还要大一点
#A2C2022-12-03-16-00-19 1.6学习过程可以 A2C2022-12-03-17-18-39.pdf 不太好
#A2C2022-12-03-16-00-44 1.8学习过程可以 A2C2022-12-03-17-18-49.pdf 可以的

#是不是价格差别还要大一点
#A2C2022-12-03-16-08-08  1.4学习过程有下降 A2C2022-12-03-17-25-37.pdf 可以的

#看一下benchmark的表现
#A2C2022-12-03-16-14-02 1.6
#A2C2022-12-03-16-14-49 1.8  以上两个说明EIB是lf越长，对于myopic的优势越大 
#A2C2022-12-03-16-15-27 1.4价格差别变大  对比不明显

#价格差变大，而且变长
#A2C2022-12-03-17-55-22
#A2C2022-12-03-17-55-25
#上面两个效果不好，好像是学习率大了
#减小初始学习率，减慢衰减
#A2C2022-12-03-21-22-56 1.6 不怎么好
#A2C2022-12-03-21-23-00 1.8 A2C2022-12-03-21-23-00.pdf很好
#A2C2022-12-03-22-41-40 还可以，优势不明显A2C2022-12-03-22-41-40.pdf
#A2C2022-12-03-22-53-28 还可以

#改成epoch
#A2C2022-12-10-22-06-08 还可以

#A2C2022-12-10-23-45-43最后用的这个

#A2C2022-12-17-16-20-47改进了sample

#用的这个A2C2022-12-17-18-44-56