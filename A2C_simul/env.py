import numpy as np
from uti import create_data
import torch
import time
class market_dynamic:
    def __init__(self,args,Resnet,seg_prob,
                 initial_inventory,products_price,T,Train=False):
        self.args = args
        self.Resnet = Resnet
        self.device = args.device
        self.seg_prob=seg_prob
        self.batch_size = args.train_batch_size if Train else args.batch_size
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.initial_inventory = np.tile(initial_inventory, (self.batch_size, 1))
        self.total_inv = initial_inventory.sum()
        self.products_price=products_price
        self.num_of_products=len(initial_inventory)
        self.cardinality = args.cardinality
        self.purchase=np.zeros((self.batch_size,self.num_of_products),dtype= np.int)
        self.total_T = T
        self.T=T#剩余的销售时间
        X, Z, _,concat_feature0 = create_data(args)
        self.products_features = X
        self.num_p_features = X.shape[1]
        self.customer_features = Z
        self.num_of_customer_segment = len(Z)
        self.concat_feature0 = concat_feature0
    def reset(self,initial_inventory,T):
        self.inventory_level = np.tile(initial_inventory, (self.batch_size, 1))
        self.T=T
    def step(self,arriving_seg,assortment):#assortment是torch.zeros([env.batch_size, self.num_products],dtype=torch.int)
        #考虑batch,生成input
        pre = time.time()
        multiplier = (np.hstack((assortment, np.ones((self.batch_size, 1))))).reshape(self.batch_size, self.num_of_products+1, 1)
        multiplier = np.concatenate((np.repeat(multiplier, 8, axis=2), np.ones((self.batch_size, self.num_of_products+1, self.args.num_cus_features))), axis=2)
        arriving_seg_feature = self.concat_feature0[arriving_seg.ravel(), :]
        input_1 = torch.from_numpy(multiplier * arriving_seg_feature).float().to(self.device)  #product feature加cus feature
        ass_onehot = torch.from_numpy(np.hstack((assortment, np.ones((self.batch_size, 1))))).float().to(self.device)
        # 关键语句
        with torch.no_grad():
            prob = self.Resnet(input_1, ass_onehot)  # input1是商品特征加上顾客特征
        # 初始化
        prices = np.hstack((self.products_price,np.array([0])))
        index = torch.multinomial(prob,1).cpu()
        self.purchase = torch.zeros((self.batch_size, self.num_of_products+1),dtype= np.int)
        self.purchase.scatter_(1,index,1)
        index = index.numpy()
        reward = prices[index]
        self.inventory_level-=self.purchase[:,:-1].numpy()#最后一列是不买
        self.T-=1
        now = time.time()
        #print('time:', now-pre)
        #breakpoint()
        return index,reward
    def get_mask(self):
        mask = self.inventory_level.copy()
        mask[self.inventory_level == 0] = 1
        mask[self.inventory_level > 0]=0
        return mask
    def all_finished(self):
        if self.T == 1:
            return True
        else:
            return False