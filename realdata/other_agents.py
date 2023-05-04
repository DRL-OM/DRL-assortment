import numpy as np
import random
from func import Cardinality_ass,prob_of_products
from uti import create_data
from arg import init_parser
import scipy.stats as st

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
        #print(self.market.inventory_level,OA_ass,reward)
        self.total_reward += reward

class myopic_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        X, Z, concat_feature,_ = create_data(args)
        self.MNL_para = MNL_para
        self.products_feature = X 
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality=args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def myopic_ass(self,arriving_seg):
        myopic_ass = []
        for i,cus in enumerate(arriving_seg):
            V = np.exp((self.products_feature @ self.MNL_para[cus[0]]).ravel())
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
        #print(self.market.inventory_level,myopic_ass,reward)
        self.total_reward += reward
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
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
        X, Z, concat_feature,_ = create_data(args)
        self.MNL_para = MNL_para
        self.products_feature = X
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
    def IB_ass(self,arriving_seg):
        IB_ass = []
        for i,cus in enumerate(arriving_seg):
            V = np.exp((self.products_feature @ self.MNL_para[cus[0]]).ravel())
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
        #print(self.market.inventory_level,IB_ass,reward)
        self.total_reward += reward

        

class sub_t_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.args = args
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        prop_features = np.load('resnet/expedia_prop_features.npy')
        self.MNL_para = np.exp(MNL_para@prop_features.T)[:,:-1]
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.t = 0
        self.segprob = get_segprob(args)
    def sub_t_ass(self,arriving_seg):
        sub_t_ass = []
        for i,cus in enumerate(arriving_seg):
            T = 130
            lambda_ = 0.9237
            d_t,alpha_t = information_t(self.MNL_para,self.market.inventory_level[0],self.segprob)
            d_t = lambda_*d_t
            D_t = variable_Dt(d_t,self.t,T)
            D_tS = variable_DtS(D_t,alpha_t,self.market.inventory_level[0])
            delta = []
            for i in range(30):
                p1 = len(D_tS[i][np.where(D_tS[i]>self.market.inventory_level[0][i])])/10000
                j_list = list(np.arange(30))
                j_list.remove(i)
                sum_ = 0
                for ind,j in enumerate(j_list):
                    p2 = len(D_tS[j][np.where(D_tS[j]<=self.market.inventory_level[0][j])])/10000
                    p3 = len(D_t[i][np.where(D_t[i]>self.market.inventory_level[0][i])])/10000
                    sum_ += alpha_t[i][ind]*p2*p3
                delta.append(self.products_price[0][i]*(p1-sum_))
            delta = np.array(delta)
            prices = np.array(self.products_price[0])-delta
            copy_inv = self.market.inventory_level[0].copy()
            copy_inv[copy_inv>0] = 1
            prices = prices*copy_inv#不摆库存为0的
            
            V = self.MNL_para[cus[0]]
            sub_t_ass.append(Cardinality_ass(V,prices,self.cardinality))
        if random.random()>lambda_ and self.t<130:
            self.t += 1 
        return np.array(sub_t_ass)
    def reset(self, initial_inventory, T, products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        self.t = 0
    def step(self,arriving_seg,check):
        sub_t_ass = self.sub_t_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, sub_t_ass,check)
        self.total_reward += reward   
        #print(self.market.inventory_level[0],sub_t_ass.nonzero()[1],reward)
        if self.t<130:
            self.t += 1   
                
        
        
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

def information_t(MNL_para,inventory_level,seg_prob):
    MNL_para_t = MNL_para.copy()
    MNL_para_t[:,np.where(inventory_level==0)] = 0
    
    V_sum = (MNL_para_t.sum(1)+1).reshape((4,1))
    p_i = MNL_para/V_sum
    d_it = (p_i*seg_prob.reshape((4,1))).sum(0)
    '''
    alpha_ijt = []
    for i in range(30):
        mnl = MNL_para.copy()
        mnl[:,i] = 0
        mnl_t = MNL_para_t.copy()
        mnl_t[:,i] = 0
        line = (mnl/(mnl_t.sum(1)+1).reshape((4,1))*seg_prob.reshape((4,1))).sum(0)
        alpha_ijt.append(line)      
    breakpoint()'''
    alpha_ijt = 0
    for i in range(4):
        mnl = np.repeat(MNL_para[i].reshape((1,30)),30,axis=0)
        mnl_t = np.repeat(MNL_para_t[i].reshape((1,30)),30,axis=0)
        row, col = np.diag_indices_from(mnl)
        mnl[row,col] = 0
        row, col = np.diag_indices_from(mnl_t)
        mnl_t[row,col] = 0
        alpha_ijt += mnl/((mnl_t.sum(1)+1).reshape((30,1)))*seg_prob[i]
    
    return d_it,alpha_ijt
    
def variable_Dt(d_t,t,T):  
    D_t=[]
    for i in range(30):
        D_t.append(st.poisson.rvs(d_t[i]*(T-t), loc=0, size=10000))
    return np.array(D_t)

def variable_DtS(D_t,alpha_t,inventory_level):
    '''
    D_tS = []
    breakpoint()
    for i in range(30):
        j_list = np.arange(30)
        sum_ = 0
        for j in j_list:
            if inventory_level[j] > 0:
                minus = D_t[j]-inventory_level[j]
                minus[minus<0] = 0
                sum_ += alpha_t[j][i]*minus#alpha_t[i][i]=0
        D_tS.append(D_t[i]+sum_)'''
    minus = D_t-inventory_level.reshape((30,1))
    minus[minus<0] = 0
    minus[inventory_level==0,:]=0
    D_tS = D_t + (alpha_t.reshape((30,30,1))*np.repeat(minus.reshape((1,30,10000)),30,axis=0)).sum(0)
    
    return D_tS

def get_segprob(args):
    people1 = 0
    people2 = 0
    people3 = 0
    people4 = 0
    lenghts = []
    max_length = 0
    for l in list(args.seqdata.values())[:160]:
        lenghts.append(len(l))
        people1+=l.count(1)
        people2+=l.count(2)
        people3+=l.count(3)
        people4+=l.count(4)
        if len(l)>max_length:
            max_length = len(l)#109
    seg_prob_ = [people1/np.sum(lenghts),people2/np.sum(lenghts),people3/np.sum(lenghts),people4/np.sum(lenghts)]
    return np.array(seg_prob_)

if __name__ == '__main__':
    import scipy.stats as st
    import json
    import pandas
    from scipy.optimize import curve_fit
    import scipy.io as io

    products_price = [1.838419913,1.992458678,1.874724518,1.515151515,3.28728191,2.695362718,1.032467532,4.485454545,1.57983683,1.02]
    '''
    #读取MNL参数
    MNL_para1 = io.loadmat('MNL/20round_beta1.mat')
    MNL_para1 = MNL_para1['var'].reshape((-1,6))
    MNL_para2 = io.loadmat('MNL/20round_beta2.mat')
    MNL_para2 = np.vstack((  MNL_para1 , MNL_para2['var'].reshape((-1,6)) ))
    MNL_para3 = io.loadmat('MNL/20round_beta3.mat')
    MNL_para3 = np.vstack((  MNL_para2 , MNL_para3['var'].reshape((-1,6)) ))
    MNL_para4 = io.loadmat('MNL/20round_beta4.mat')
    MNL_para = np.vstack((  MNL_para3 , MNL_para4['var'].reshape((-1,6)) ))
    prop_features = np.load('resnet/expedia_prop_features.npy')
    MNL_para = np.exp(MNL_para@prop_features.T)[:,:-1]
    initial_inventory = np.array([2]*30)
    
    with open('../../expedia/new/seqdata.json', 'r') as f:
        seqdata = json.loads(f.read())
    people1 = 0
    people2 = 0
    people3 = 0
    people4 = 0
    lenghts = []
    max_length = 0
    for l in list(seqdata.values())[:160]:
        lenghts.append(len(l))
        people1+=l.count(1)
        people2+=l.count(2)
        people3+=l.count(3)
        people4+=l.count(4)
        if len(l)>max_length:
            max_length = len(l)#109
    seg_prob_ = [people1/np.sum(lenghts),people2/np.sum(lenghts),people3/np.sum(lenghts),people4/np.sum(lenghts)]
    breakpoint()
    T = 130
    lambda_ = 0.9237
    
    d_0,alpha_0 = information_t(MNL_para,initial_inventory,seg_prob)
    d_0 = lambda_*d_0
    D_0 = variable_Dt(d_0,0,T)
    D_0S = variable_DtS(D_0,alpha_0,initial_inventory)
    delta = []
    for i in range(10):
        p1 = len(D_0S[i][np.where(D_0S[i]>initial_inventory[i])])/1000000
        j_list = [0,1,2,3,4,5,6,7,8,9]
        j_list.remove(i)
        sum_ = 0
        for ind,j in enumerate(j_list):
            p2 = len(D_0S[j][np.where(D_0S[j]<=initial_inventory[j])])/1000000
            p3 = len(D_0[i][np.where(D_0[i]>initial_inventory[i])])/1000000
            sum_ += alpha_0[i][ind]*p2*p3
        delta.append(products_price[i]*(p1-sum_))
    delta = np.array(delta)
    prices = np.array(products_price)-delta
    breakpoint()'''