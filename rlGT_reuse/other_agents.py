import numpy as np
import random
from func import Cardinality_ass,prob_of_products
from uti import create_data
from arg import init_parser

class OA_agent:
    def __init__(self,args,env_,products_price):
        self.market=env_
        self.batch_size = args.batch_size
        self.cardinality = args.cardinality
        self.products_price=products_price
        self.total_reward=np.zeros((self.batch_size,1))
        self.N=len(products_price)
    def OA(self):
        # 随机选cardinality个商品
        ass = np.zeros((self.batch_size,self.N))
        range_ = self.market.inventory_level.nonzero()[1]
        for i in range(self.batch_size):
            try:
                random_choose = random.sample(list(range_), random.randint(1,self.cardinality))
            except:
                random_choose = random.sample(list(range_), random.randint(1,len(range_)))
            ass[i][random_choose] = 1
        ass = ass*self.market.inventory_level
        ass[ass>0] = 1
        return ass
    def reset(self,initial_inventory,T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg,check):
        OA_ass = self.OA()
        _,reward = self.market.step(arriving_seg, OA_ass,check)
        self.total_reward += reward

class myopic_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.MNL_para = MNL_para
        self.p = products_price
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality=args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def myopic_ass(self,arriving_seg):
        myopic_ass = []
        for i,cus in enumerate(arriving_seg):
            V = self.MNL_para[cus[0]]
            myopic_ass.append(
                Cardinality_ass(V,self.products_price[i],self.cardinality))
        return np.array(myopic_ass)
    def reset(self,initial_inventory,T,products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg,check):
        #self.exam(arriving_seg)
        myopic_ass = self.myopic_ass(arriving_seg)
        choose_index,reward = self.market.step(arriving_seg, myopic_ass,check)
        self.total_reward += reward
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
        self.products_price = np.tile(self.p,(self.batch_size,1))
        self.products_price = self.products_price*copy_inv#不摆库存为0的
    def exam(self,arriving_seg):
        #检验真实和虚假的V的差别
        breakpoint()
        V = self.market.Vs[arriving_seg]
        guess_V = np.random.random(20)*2
        myopic_ass = Cardinality_ass(V,self.products_price,self.cardinality)
        F_myopic_ass = Cardinality_ass(guess_V,self.products_price,self.cardinality)
        Re = prob_of_products(myopic_ass, V)[:-1]@self.products_price
        F_Re = prob_of_products(F_myopic_ass, V)[:-1]@self.products_price

def E_penalty_function(x):
    return (1-np.exp(-x))*(np.e/(np.e-1))
def L_penalty_function(x):
    return x

class E_IB_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = MNL_para
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def IB_ass(self,arriving_seg):
        IB_ass = []
        for i,cus in enumerate(arriving_seg):
            V = self.MNL_para[cus[0]]
            r_ = E_penalty_function(self.market.inventory_level[i]/
                                    self.initial_inventory[i]) \
                                    * self.products_price[i]
            IB_ass.append(Cardinality_ass(V,r_,self.cardinality))
        return np.array(IB_ass)
    def reset(self, initial_inventory, T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg,check):
        IB_ass = self.IB_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, IB_ass,check)
        self.total_reward += reward
        
from itertools import combinations
class optmyopic_agent:
    def __init__(self, args, env_, products_price):
        self.args = args
        self.rank_list = args.rank_list
        self.cus_type = args.cus_type
        self.market = env_
        self.products_price = products_price
        self.total_reward = 0
        self.N = len(products_price)
        self.cardinality = args.cardinality
        self.S = np.zeros((10+45+120+210,10))
        index = 0
        for i in range(1, self.cardinality+1):
            for a in combinations(range(self.N), i):
                self.S[index,list(a)] = 1
                index+=1
    def optmyopic_ass(self,arriving_seg):
        best_reward = 0
        best_ass = 0
        for ass_ind in range(len(self.S)):
            ass = self.S[ass_ind]
            reward = self.products_price@prob_of_products_rl(self.rank_list,self.cus_type,arriving_seg,ass)[:-1]
            if reward > best_reward:
                best_reward = reward
                best_ass = ass
        return np.array([best_ass])
    def reset(self,initial_inventory, T, products_price):
        self.market.reset(initial_inventory, T)
        self.products_price = products_price
        self.total_reward = 0
    def step(self,arriving_seg):
        ass = self.optmyopic_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, ass)
        self.total_reward += reward[0][0]
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
        self.products_price = self.products_price*copy_inv[0]#不摆库存为0的

class optEIB_agent:
    def __init__(self, args, env_, products_price):
        self.args = args
        self.rank_list = args.rank_list
        self.cus_type = args.cus_type
        self.market = env_
        self.initial_inventory = env_.initial_inventory
        self.products_price = products_price
        self.total_reward = 0
        self.N = len(products_price)
        self.cardinality = args.cardinality
        self.S = np.zeros((10+45+120+210,10))
        index = 0
        for i in range(1, self.cardinality+1):
            for a in combinations(range(self.N), i):
                self.S[index,list(a)] = 1
                index+=1
    def optEIB_ass(self,arriving_seg):
        best_reward = 0
        best_ass = np.zeros([1,10])
        for ass_ind in range(len(self.S)):
            ass = self.S[ass_ind]
            r_ = E_penalty_function(self.market.inventory_level[0]/
                                    self.initial_inventory[0]) \
                                    * self.products_price
            reward = r_@prob_of_products_rl(self.rank_list,self.cus_type,arriving_seg,ass)[:-1]
            if reward > best_reward:
                best_reward = reward
                best_ass = ass
        return np.array([best_ass])
    def reset(self,initial_inventory, T):
        self.market.reset(initial_inventory, T)
        self.total_reward = 0
    def step(self,arriving_seg):
        ass = self.optEIB_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, ass)
        self.total_reward += reward[0][0]
        #print(self.market.T,self.market.inventory_level,ass,choose_index,self.total_reward)
        
        
def prob_of_products_rl(rank_list,cus_type,arriving_seg,assort_onehot):
    arriving_cus = cus_type[arriving_seg].reshape(1,100)[0]#每一行是这个cus type的list概率
    ass = np.hstack((assort_onehot,1))
    ass = np.repeat(np.arange(1,12).reshape(1,11),1,0)*ass
    ass = ass.reshape(1,11,1)
    prob_list = np.zeros(11)
    for i in range(len(rank_list)):
        choose = rank_list[i][np.min(np.where(rank_list[i]==ass[0])[-1])]#在这个list里面会选什么
        prob_list[choose-1] += arriving_cus[i]
    return prob_list








