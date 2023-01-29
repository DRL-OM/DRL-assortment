import numpy as np
import random
from func import Cardinality_ass,prob_of_products
from uti import create_data
from arg import init_parser
parser = init_parser('Reinforce')
args = parser.parse_args()
X, Z, concat_feature,_ = create_data(args)

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
        for i in range(self.batch_size):
            random_choose = random.sample(range(self.N), self.cardinality)
            ass[i][random_choose] = 1
        ass = ass*self.market.inventory_level
        ass[ass>0] = 1
        return ass
    def reset(self,initial_inventory,T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg):
        OA_ass = self.OA()
        _,reward = self.market.step(arriving_seg, OA_ass)
        self.total_reward += reward

class myopic_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.MNL_para = MNL_para
        self.products_feature = concat_feature
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality=args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def myopic_ass(self,arriving_seg):
        myopic_ass = []
        for i,cus in enumerate(arriving_seg):
            V = np.exp((self.products_feature[cus] @ self.MNL_para).ravel())
            myopic_ass.append(
                Cardinality_ass(V,self.products_price[i],self.cardinality))
        return np.array(myopic_ass)
    def reset(self,initial_inventory,T,products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg):
        #self.exam(arriving_seg)
        myopic_ass = self.myopic_ass(arriving_seg)
        choose_index,reward = self.market.step(arriving_seg, myopic_ass)
        self.total_reward += reward
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
        self.products_price = self.products_price*copy_inv
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
        self.products_feature = concat_feature
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def IB_ass(self,arriving_seg):
        IB_ass = []
        for i,cus in enumerate(arriving_seg):
            V = np.exp((self.products_feature[cus] @ self.MNL_para).ravel())
            r_ = E_penalty_function(self.market.inventory_level[i]/
                                    self.initial_inventory[i]) \
                                    * self.products_price[i]
            IB_ass.append(Cardinality_ass(V,r_,self.cardinality))
        return np.array(IB_ass)
    def reset(self, initial_inventory, T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg):
        IB_ass = self.IB_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, IB_ass)
        self.total_reward += reward
'''
class L_IB_agent:
    def __init__(self, env_, products_price, initial_inventory):
        self.market = env_
        self.products_price = products_price
        self.initial_inventory = initial_inventory
        self.total_reward = 0
        self.N = len(products_price)
    def IB_ass(self,arriving_seg):
        V=self.market.Vs[arriving_seg]
        r_ = L_penalty_function(self.market.inventory_level/self.initial_inventory) * self.products_price
        IB_ass=get_myopic_ass(r_,V,self.market.inventory_level)
        return IB_ass
    def reset(self,initial_inventory,T):
        self.market.reset(initial_inventory,T)
        self.total_reward = 0
    def step(self,arriving_seg):
        IB_ass = self.IB_ass(arriving_seg)
        reward = self.market.step(arriving_seg, IB_ass)
        self.total_reward += reward
'''

'''from itertools import combinations
class opt_agent:
    def __init__(self, env_, products_price):
        self.market = env_
        self.products_price = products_price
        self.total_reward = 0
        self.N = len(products_price)
        self.S = np.zeros((31,5))
        index = 0
        for i in range(1, self.N+1):
            for a in combinations(range(self.N), i):
                self.S[index,list(a)] = 1
                index+=1
    def opt_ass(self,arriving_seg):
        pref=self.market.customer_preferences[arriving_seg]
        myopic_ass=get_myopic_ass(self.products_price,pref,self.market.products_feature,self.market.inventory_level)
        return myopic_ass
    def income(self,purchase_index):
        if purchase_index != -1:
            reward=self.products_price[purchase_index]
            self.total_reward+=reward
    def reset(self,initial_inventory):
        self.market.reset(initial_inventory)
        self.total_reward = 0
    def step(self,arriving_seg):
        myopic_ass = self.myopic_ass(arriving_seg)
        myopic_purchase = self.market.customer_choose(arriving_seg, myopic_ass)
        self.income(myopic_purchase)
        self.market.inventory_change()'''
