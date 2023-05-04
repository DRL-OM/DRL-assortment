import numpy as np
import random
from func import Cardinality_ass,prob_of_products,exp_rev
from uti import create_data
from arg import init_parser
import scipy.stats as st
import cvxpy as cp
import json

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
        self.MNL_para = np.exp(MNL_para)
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
        self.MNL_para = np.exp(MNL_para)
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
            #print('IB_ass:',r_,np.array(IB_ass).nonzero()[1])
        return np.array(IB_ass)
    def reset(self, initial_inventory, T):
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
    def step(self,arriving_seg,check):
        IB_ass = self.IB_ass(arriving_seg)
        choose_index, reward = self.market.step(arriving_seg, IB_ass,check)
        self.total_reward += reward
        #print('IB_ass_reward:',self.total_reward)

class sub_t_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.args = args
        self.market = env_
        self.batch_size = args.batch_size
        self.initial_inventory = env_.initial_inventory
        self.MNL_para = np.exp(MNL_para)
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality = args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.t = 0
        self.segprob = get_segprob(args)
    def sub_t_ass(self,arriving_seg):
        sub_t_ass = []
        for i,cus in enumerate(arriving_seg):
            T = 109
            lambda_ = 0.916
            d_t,alpha_t = information_t(self.MNL_para,self.market.inventory_level[0],self.segprob)
            d_t = lambda_*d_t
            D_t = variable_Dt(d_t,self.t,T)
            D_tS = variable_DtS(D_t,alpha_t,self.market.inventory_level[0])
            delta = []
            for i in range(10):
                p1 = len(D_tS[i][np.where(D_tS[i]>self.market.inventory_level[0][i])])/10000
                j_list = [0,1,2,3,4,5,6,7,8,9]
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
            #print('sub_t_ass:',prices,np.array(sub_t_ass).nonzero()[1])
        if random.random()>lambda_ and self.t<109:
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
        #print('sub_t_reward:',self.total_reward)
        if self.t<109:
            self.t += 1   

class LP_agent:
    def __init__(self,args, env_, MNL_para, products_price):
        self.market = env_
        self.batch_size = args.batch_size
        self.MNL_para = np.exp(MNL_para)
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.cardinality=args.cardinality
        self.total_reward = np.zeros((self.batch_size,1))
        self.N = len(products_price)
        self.SS = 0
        self.primal = 0
        self.seg_prob_ = get_seg()
    def LP_ass(self,arriving_seg):
        for i,cus in enumerate(arriving_seg):
            choose_prob = self.primal[cus[0]]/np.sum(self.primal[cus[0]])
            try:
                choice = np.random.choice(np.arange(len(choose_prob)), 1, p = choose_prob)
            except:
                choice = np.array([0])
        return np.array([self.SS[cus[0]][int(choice)]])
    def reset(self,initial_inventory,T,products_price):
        self.products_price = np.tile(products_price,(self.batch_size,1))
        self.market.reset(initial_inventory,T)
        self.total_reward = np.zeros((self.batch_size,1))
        self.SS = [[[1,1,1,1,0,0,0,0,0,0]],[[1,1,1,1,0,0,0,0,0,0]],[[1,1,1,1,0,0,0,0,0,0]],[[1,1,1,1,0,0,0,0,0,0]]]
        add = True
        while add:
            beta_dual = dual(self.SS,self.seg_prob_,self.products_price[0],self.MNL_para,self.market.inventory_level[0])
            self.SS,add = column_gene(self.products_price[0],beta_dual,self.SS,self.MNL_para,self.seg_prob_)
        self.primal = primal(self.SS,self.seg_prob_,self.products_price[0],self.MNL_para,self.market.inventory_level[0])
    def step(self,arriving_seg,check):
        #self.exam(arriving_seg)
        LP_ass = self.LP_ass(arriving_seg)
        choose_index,reward = self.market.step(arriving_seg, LP_ass,check)
        self.total_reward += reward
        copy_inv = self.market.inventory_level.copy()
        copy_inv[copy_inv>0] = 1
        if (copy_inv<0).any():
            raise NotImplementedError
        if (self.products_price*copy_inv != self.products_price).any():
            self.products_price = self.products_price*copy_inv#不摆库存为0的
            self.SS = [[[0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0]]]
            one = list(self.market.inventory_level[0]>0).index(1)
            for ss in self.SS:
                ss[0][one]=1
            add = True
            while add:
                beta_dual = dual(self.SS,self.seg_prob_,self.products_price[0],self.MNL_para,self.market.inventory_level[0])
                self.SS,add = column_gene(self.products_price[0],beta_dual,self.SS,self.MNL_para,self.seg_prob_)
            self.primal = primal(self.SS,self.seg_prob_,self.products_price[0],self.MNL_para,self.market.inventory_level[0])
        
        
        
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


def information_t(MNL_para,inventory_level,seg_prob):
    MNL_para_t = MNL_para.copy()
    MNL_para_t[:,np.where(inventory_level==0)] = 0
    
    V_sum = (MNL_para_t.sum(1)+1).reshape((4,1))
    p_i = MNL_para/V_sum
    d_it = (p_i*seg_prob.reshape((4,1))).sum(0)
    '''
    alpha_ijt = []
    for i in range(10):
        mnl = MNL_para.copy()
        mnl[:,i] = 0
        mnl_t = MNL_para_t.copy()
        mnl_t[:,i] = 0
        line = (mnl/(mnl_t.sum(1)+1).reshape((4,1))*seg_prob.reshape((4,1))).sum(0)
        alpha_ijt.append(line)'''    
    alpha_ijt = 0
    for i in range(4):
        mnl = np.repeat(MNL_para[i].reshape((1,10)),10,axis=0)
        mnl_t = np.repeat(MNL_para_t[i].reshape((1,10)),10,axis=0)
        row, col = np.diag_indices_from(mnl)
        mnl[row,col] = 0
        row, col = np.diag_indices_from(mnl_t)
        mnl_t[row,col] = 0
        alpha_ijt += mnl/((mnl_t.sum(1)+1).reshape((10,1)))*seg_prob[i]
    
    return d_it,alpha_ijt
    
def variable_Dt(d_t,t,T):  
    D_t=[]
    for i in range(10):
        D_t.append(st.poisson.rvs(d_t[i]*(T-t), loc=0, size=10000))
    return np.array(D_t)

def variable_DtS(D_t,alpha_t,inventory_level):
    '''
    D_tS = []
    for i in range(10):
        j_list = np.array([0,1,2,3,4,5,6,7,8,9])
        sum_ = 0
        for j in j_list:
            if inventory_level[j] > 0:
                minus = D_t[j]-inventory_level[j]
                minus[minus<0] = 0
                sum_ += alpha_t[j][i]*minus#alpha_t[i][i]=0
        D_tS.append(D_t[i]+sum_)
    return np.array(D_tS)'''
    minus = D_t-inventory_level.reshape((10,1))
    minus[minus<0] = 0
    minus[inventory_level==0,:]=0
    D_tS = D_t + (alpha_t.reshape((10,10,1))*np.repeat(minus.reshape((1,10,10000)),10,axis=0)).sum(0)
    
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
        people1+=l.count(0)
        people2+=l.count(1)
        people3+=l.count(2)
        people4+=l.count(3)
        if len(l)>max_length:
            max_length = len(l)#109
    seg_prob_ = [people1/np.sum(lenghts),people2/np.sum(lenghts),people3/np.sum(lenghts),people4/np.sum(lenghts)]
    return np.array(seg_prob_)

def primal(SS,seg_prob_,products_price,MNL_para,inventory_level):
    num_var = len(SS[0])+len(SS[1])+len(SS[2])+len(SS[3])
    vars_ = cp.Variable(num_var)
    v = 0
    constraints=[]
    Rev=cp.Constant(0)
    num = 0
    for z,S in zip(range(4),SS):
        constraints.append(cp.sum(vars_[num:num+len(S)])<=seg_prob_[z])
        for s in S:
            Rev += seg_prob_[z]*exp_rev(np.array(s),MNL_para[z],products_price)*vars_[v]
            v += 1
        num += len(S)
    for i in range(10):
        num = 0
        sum_ = 0
        for z,S in zip(range(4),SS):
            prob_list = []
            for s in S:
                prob_list.append(seg_prob_[z]*prob_of_products(np.array(s),MNL_para[z])[i])
            sum_ += prob_list@vars_[num:num+len(S)]
            num += len(S)
        constraints.append(sum_<=inventory_level[i])
    constraints.append(0 <= vars_)
    objective=cp.Maximize(Rev)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    primal1 = np.array(beta[:len(SS[0])])
    primal1[primal1<0] = 0
    primal2 = np.array(beta[len(SS[0]):len(SS[0])+len(SS[1])])
    primal2[primal2<0] = 0
    primal3 = np.array(beta[len(SS[0])+len(SS[1]):len(SS[0])+len(SS[1])+len(SS[2])])
    primal3[primal3<0] = 0
    primal4 = np.array(beta[len(SS[0])+len(SS[1])+len(SS[2]):])
    primal4[primal4<0] = 0
    primal = [primal1,primal2,primal3,primal4]
    return primal


def dual(SS,seg_prob_,products_price,MNL_para,inventory_level):
    vars_ = cp.Variable(14)
    obj = cp.Constant(0)
    constraints=[]
    obj += vars_[:10]@inventory_level
    obj += vars_[10:]@seg_prob_
    for z,S in zip(range(4),SS):
        for s in S:
            left = seg_prob_[z]*(vars_[:10]@prob_of_products(np.array(s),MNL_para[z])[:-1])
            right = seg_prob_[z]*(products_price@prob_of_products(np.array(s),MNL_para[z])[:-1])
            constraints.append(right <= vars_[z+10]+left)
    constraints.append(0 <= vars_)
    objective=cp.Minimize(obj)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    return beta

def column_gene(products_price,beta_dual,SS,MNL_para,seg_prob_):
    r_ = products_price-beta_dual[:10]
    obj_max = 0
    add = True
    for z in range(4):
        ass_ = Cardinality_ass(MNL_para[z],r_,4)
        obj_ = seg_prob_[z]*exp_rev(ass_,MNL_para[z],r_)-beta_dual[z+10]
        if obj_ > obj_max:
            obj_max = obj_
            z_max = z
            ass_max = ass_
    if obj_max == 0:
        add = False
    else:
        SS[z_max].append(list(ass_max))
    return SS,add

def get_seg():
    with open('GT/seqdata.json', 'r') as f:
        seqdata = json.loads(f.read())
    people1 = 0
    people2 = 0
    people3 = 0
    people4 = 0
    lenghts = []
    for l in list(seqdata.values())[:160]:
        lenghts.append(len(l))
        people1+=l.count(0)
        people2+=l.count(1)
        people3+=l.count(2)
        people4+=l.count(3)
    seg_prob_ = np.array([people1/np.sum(lenghts),people2/np.sum(lenghts),people3/np.sum(lenghts),people4/np.sum(lenghts)])
    seg_prob_ = seg_prob_*np.mean(lenghts)
    return seg_prob_
    
if __name__ == '__main__':
    import scipy.stats as st
    import json
    import pandas
    from scipy.optimize import curve_fit
    import cvxpy as cp

    products_price = np.array([1.838419913,1.992458678,1.874724518,1.515151515,3.28728191,2.695362718,1.032467532,4.485454545,1.57983683,1.02])
    MNL_para = np.exp(np.load('GT/Learned_MNL_weight.npy'))
    initial_inventory = np.array([10]*10)
    seg_prob = np.load('GT/cusseg_prob.npy')
    
    with open('GT/seqdata.json', 'r') as f:
        seqdata = json.loads(f.read())
    people1 = 0
    people2 = 0
    people3 = 0
    people4 = 0
    lenghts = []
    max_length = 0
    for l in list(seqdata.values())[:160]:
        lenghts.append(len(l))
        people1+=l.count(0)
        people2+=l.count(1)
        people3+=l.count(2)
        people4+=l.count(3)
        if len(l)>max_length:
            max_length = len(l)#109
    seg_prob_ = np.array([people1/np.sum(lenghts),people2/np.sum(lenghts),people3/np.sum(lenghts),people4/np.sum(lenghts)])
    seg_prob_ = seg_prob_*np.mean(lenghts)
    S1 = [[0,1,1,0,0,1,0,0,0,0]]
    S2 = [[0,1,1,0,0,1,0,0,0,0]]
    S3 = [[0,1,1,0,0,1,0,0,0,0]]
    S4 = [[0,1,1,0,0,1,0,0,0,0]]
    SS = [S1,S2,S3,S4]
    '''
    num_var = len(S1)+len(S2)+len(S3)+len(S4)
    vars_ = cp.Variable(num_var)
    v = 0
    constraints=[]
    Rev=cp.Constant(0)
    num = 0
    for z,S in zip(range(4),[S1,S2,S3,S4]):
        constraints.append(cp.sum(vars_[num:num+len(S)])<=seg_prob_[z])
        for s in S:
            Rev += seg_prob_[z]*exp_rev(np.array(s),MNL_para[z],products_price)*vars_[v]
            v += 1
        num += len(S)
    for i in range(10):
        num = 0
        sum_ = 0
        for z,S in zip(range(4),[S1,S2,S3,S4]):
            prob_list = []
            for s in S:
                prob_list.append(seg_prob_[z]*prob_of_products(np.array(s),MNL_para[z])[i])
            sum_ += prob_list@vars_[num:num+len(S)]
            num += len(S)
        constraints.append(sum_<=initial_inventory[i])
    constraints.append(0 <= vars_)
    
    objective=cp.Maximize(Rev)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    print(beta)'''
    
    vars_ = cp.Variable(14)
    obj = cp.Constant(0)
    constraints=[]
    obj += vars_[:10]@initial_inventory
    obj += vars_[10:]@seg_prob_
    for z,S in zip(range(4),[S1,S2,S3,S4]):
        for s in S:
            left = seg_prob_[z]*(vars_[:10]@prob_of_products(np.array(s),MNL_para[z])[:-1])
            right = seg_prob_[z]*(products_price@prob_of_products(np.array(s),MNL_para[z])[:-1])
            constraints.append(right <= vars_[z+10]+left)
    constraints.append(0 <= vars_)
    objective=cp.Minimize(obj)
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    print(beta)
    r_ = products_price-beta[:10]
    
    obj_max = 0
    for z in range(4):
        ass_ = Cardinality_ass(MNL_para[z],r_,4)
        obj_ = seg_prob_[z]*exp_rev(ass_,MNL_para[z],r_)-beta[z+10]
        if obj_ > obj_max:
            obj_max = obj_
            z_max = z
            ass_max = ass_
    SS[z_max].append(list(ass_max))
    breakpoint()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
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
    
    
    
    
    





