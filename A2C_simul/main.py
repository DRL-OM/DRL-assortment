from feature import encoder,Res_Assort_Net,simulator
from uti import setup_logger,generate_set
from net import A2C
import torch
import numpy as np
from numpy import *
from arg import init_parser
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
#读取MNL参数
if args.num_products==100:
    data=io.loadmat(r'../Gene_databeta/p100c4/R100gene_data'+str(args.MNLdata)+'_beta.mat')
else:
    data=io.loadmat(r'../Gene_databeta/Mat/Rgene_data'+str(args.MNLdata)+'_beta.mat')
MNL_para = data['beta'].ravel()
#读取simulator数据
'''product_encoder = encoder(8,2,40,20).to(device)
cus_encoder = encoder(6,2,40,20).to(device)
net = Res_Assort_Net(22, 1, 22, 2).to(device)
torch.save(product_encoder, 'product_encoder_simul0.pth')
torch.save(cus_encoder, 'cus_encoder_simul0.pth')
torch.save(net, 'net_simul0.pth')
breakpoint()'''
if args.num_products==100:
    product_encoder = torch.load(r'../Gene_databeta/p100c4/R100product_encoder_simul.pth')#,map_location='cpu'
    cus_encoder = torch.load(r'../Gene_databeta/p100c4/R100cus_encoder_simul.pth')
    Res_Assort_Net_ =  torch.load(r'../Gene_databeta/p100c4/R100net_simul.pth')
else:
    product_encoder = torch.load(r'../Gene_databeta/simul/Rproduct_encoder_simul.pth')#,map_location='cpu'
    cus_encoder = torch.load(r'../Gene_databeta/simul/Rcus_encoder_simul.pth')
    Res_Assort_Net_ =  torch.load(r'../Gene_databeta/simul/Rnet_simul.pth')
ResNet = simulator(product_encoder, cus_encoder, Res_Assort_Net_).float().to(device)
#设定实验参数
if args.num_products==100:
    args.p_file = '../Gene_databeta/p100c4/p100_random.npy'
    args.c_file = '../Gene_databeta/p100c4/c4_random.npy'
X_0 = np.load(args.p_file)# 包括了不选
X = X_0[:-1,:]
if args.num_products==100:
    products_price = (X[:,0]*50+1000)
else:
    products_price = (X[:,0]*50+300)
times = products_price.max()
products_price = products_price/times
logger.info("products price: {}".format(products_price*times))

np.random.seed(0)
seg_prob = np.random.dirichlet(np.ones(4), size=1)[0]
args.seg_prob = seg_prob
logger.info("seg_prob {}".format(seg_prob))

initial_inventory = np.array([args.ini_inv]*args.num_products)
T = args.selling_length
logger.info("Load Factor {}".format(T/np.sum(initial_inventory)))


from train import train,test,plot_box
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
        OA_list,myopic_list,E_IB_list,seller_list = test(ResNet,MNL_para,seg_prob,initial_inventory,T,products_price,args, val_set, logger,load=True,plot= False)
            #logger.info("OA:{:.4f},myopic:{:.4f},EIB:{:.4f}".format(mean(OA_list),mean(myopic_list),mean(E_IB_list)))
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
    if not args.test_benchmark:    
        plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
                 E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))
else:
    val_set,test_set = generate_set(args,T,0,seg_prob)
    train(args,ResNet,seg_prob,products_price,initial_inventory,val_set,logger,MNL_para)
    
    OA_list_mean=np.zeros(args.test_episode)
    myopic_list_mean=np.zeros(args.test_episode)
    E_IB_list_mean=np.zeros(args.test_episode)
    seller_list_mean=np.zeros(args.test_episode)
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #读取sequence数据
        val_set,test_set = generate_set(args,T,seed_,seg_prob)
        OA_list,myopic_list,E_IB_list,seller_list = test(ResNet,MNL_para,seg_prob,
            initial_inventory,T,products_price,args, test_set, logger,load=True,plot= False)
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
        
    plot_box(args, name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
             E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))



#A2C2022-11-20-10-25-48：benchmark在val_set上的表现

    
#A2C2022-11-17-11-45-19: inv除以10
#A2C2022-11-17-11-46-04: inv除以total
#A2C2022-11-17-12-07-46: inv除以10，用cus_fea
# inv除以total，用cus_fea

#A2C2022-11-18-11-59-55: share_lr=0.001, actor_lr=0.001, critic_lr=0.01 训练了一百次，但是最后保存的模型是33次时候保存的
#A2C2022-11-18-21-45-01: 跟上面只有学习率不同，也训练了挺多次
#A2C2022-11-19-11-10-04: 直接加载上面的模型，不训练，看稳定性怎么样
#A2C2022-11-19-14-36-07: 接着上面的训练，学习率降一点

#下面都用cus_onehot
#A2C2022-11-19-16-13-18: 用cus_onehot，中间降学习率
#下面只用product encoder，这样网络的学习曲线更明显一点
#A2C2022-11-19-17-36-17: 用cus_onehot，中间降学习率
#A2C2022-11-19-16-32-41: 跟上面一样的，只是学习率都是0
#A2C2022-11-19-20-30-54: 接着上面的训练,学习率下降一点_（没有下面的好）
#A2C2022-11-19-20-39-28: 接着41训练，学习率下降多一点，__
    
#A2C2022-11-19-22-59-33: 接上面的设定，训练一个连续的,更正了学习率变化的问题，上面根本就没有除以10，并且把下降的步子变小了0.9999，batch变到了50    
#A2C2022-11-19-23-05-41: 0.999 #学到后面学习率基本为0了
#A2C2022-11-19-23-29-53: 0.99  #不行

#A2C2022-11-20-10-14-31: 0.99999
    
#发现一个问题，每一次train都重新initialize了optimizer的学习率    
#发现重大问题，每次epoch开始都重新初始化网络了!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#A2C2022-11-20-15-21-26: 0.9999                          降得太快了
#A2C2022-11-20-15-21-49: 0.999 不要
#A2C2022-11-20-15-23-49: 0.99 不要
#发现lr下降很快，第一个epoch还没完就已经-34了
#A2C2022-11-20-15-57-34：0.99999好像还不错                   降得太快了，后面表现不好
#A2C2022-11-20-16-17-54: 0.999999从这里开始不要decrease了

#A2C2022-11-20-17-01-12： 0.999999训练一次就val一次

#希望前面的差一点，更能凸显学习过程
#A2C2022-11-20-23-16-26：初始学习率低一点，下降慢一点
#A2C2022-11-20-23-43-27：初始比上面这个高10倍
#A2C2022-11-21-00-10-44: 初始低，但是不下降，好了就降0.5倍
#以上三个都表现很好！！！！！！！！！！！！！！！！！！！！！！！！！！有学习曲线！！！！！！！！！！

#A2C2022-11-21-10-42-59：看value function的变化趋势

#发现问题，商品特征和顾客特征放反了！！！！！！！！！！！！！！！self.purchase弄错了
#重新生成了mnl参数
#A2C2022-11-24-14-35-28：跑之前表现好的（用的random data: products10 cus4）

#还是用MCCM生成的数据学习的Net（带1的，prod10,c4）
#更正错误之后，差距变小了，MNL最好，Random第二了
#A2C2022-11-24-20-39-52: 这是重新跑的   到30回合左右表现很好，但是后面边很差了
#A2C2022-11-24-21-10-18.pdf这是上面的结果
#A2C2022-11-24-21-33-45：衰减大一点  大太多了
#A2C2022-11-24-21-49-50：稍微大一点        很棒！！！！结果在这里A2C2022-11-24-23-13-09.pdf

#试不同的load factor
#A2C2022-11-24-23-22-12 0.8 不错子 结果A2C2022-11-25-11-26-02.pdf
#A2C2022-11-24-23-46-49 0.6 也很好 结果A2C2022-11-25-11-27-17.pdf 比较接近
#A2C2022-11-25-10-00-42 0.5 A2C2022-11-25-11-15-17.pdf  没学好
#A2C2022-11-25-17-27-28 0.5 其实中间学到东西了，让学习率下降快一点  很好 A2C2022-11-25-18-22-52.pdf
#A2C2022-11-25-10-02-34 1.0 要检验一下1.0时候代码还是否正确（对的）  好像很好 但是结果不怎么样 A2C2022-11-25-18-03-51.pdf
#1.0的时候其实还有很多东西没卖完
#A2C2022-11-26-14-31-07 1.0 结果很好A2C2022-11-26-15-16-20.pdf
#A2C2022-11-26-15-23-15 1.1 A2C2022-11-26-16-05-10.pdf 非常好

#试不同的seg_prob
#A2C2022-11-26-16-39-01 很好 A2C2022-11-26-16-55-55.pdf

#再跑几组0.8的，画方差图
#['A2C2022-11-24-23-22-12','A2C2022-11-28-13-34-30','A2C2022-11-28-14-01-11','A2C2022-11-28-15-05-11']
#A2C2022-11-28-16-07-35.pdf对应A2C2022-11-28-15-05-11

#试随机生成网络做实验：波动很大
#A2C2022-11-29-15-21-17 A2C2022-11-29-16-06-13.pdf  很接近 学习率衰减到-10
#接近的原因是顾客差别不大
#启示：顾客差别越大，我们的方法优势会不会越明显呢？？？？？



#cus_segment改成狄利克雷分布
#用random data 
#A2C2022-12-08-16-54-50不错 lr 0.0001 0.00001 min0.00001 0.999

#A2C2022-12-08-23-13-25 非常棒！
#12/9 画了方差图 不同load factor的图
#改了图的画法

#12/9 22:40 发现了MNL beta的错误 所有实验都要重跑

#A2C2022-12-10-11-58-54 1.0  A2C2022-12-10-12-07-38 1.1 很好，可以直接用

#A2C2022-12-10-11-55-08 0.5的跑好了

#学习率太大了load factor大的曲线不好  看一下怎么调一个大家都跑的好的学习率
#目前0.8和0.6的并不完美 A2C2022-12-10-11-27-59  A2C2022-12-10-16-16-08

#画方差图 看是用不同的seed还是net seed:   用不同的net seed画出来的图好看，前面波动大后面波动小



#改成100个商品，要改哪？？？
#1.读的商品顾客文件 2.读的MNL参数 3.resnet网络 4.

#100个商品的  学习率初始应该在0.001 下降要慢

#A2C2022-12-10-21-17-13 从118升到128就不动了
#A2C2022-12-11-11-22-56 前面一直在上升

#A2C2022-12-11-17-18-30 C=8  A2C2022-12-11-17-18-30 C=4 benchmark
#A2C2022-12-11-17-22-02 92学到了105  A2C2022-12-11-22-34-58  92学到了100    A2C2022-12-11-21-32-18后面下降了
#A2C2022-12-11-17-23-56 100学到了106 

#100product 要0.01 0.001 不能两个都0.001

#A2C2022-12-12-15-38-56  A2C2022-12-12-15-40-05  benchmark    
#A2C2022-12-13-10-17-57还不错

#A2C2022-12-14-11-18-44 100好的

#A2C2022-12-15-12-24-45 1.0 重新跑的 好 0.01 0.0001

#0.01 0.0001 对于捏的来说学习率太大了

#换为放回的抽样，这样才是理论的概率
#A2C2022-12-17-10-45-40 A2C2022-12-17-11-26-56跑的1.0的 学习曲线不错，优势有待加强

#还是要用greedy去测试
