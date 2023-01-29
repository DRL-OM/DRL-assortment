import numpy as np
import random
from itertools import  permutations
num_p = 11#有一个是不买
num_cus_type = 4
seg_prob = [0.4,0.3,0.1,0.2]
num_pf = 8
num_cf = 6
card = 4
np.random.seed(0)
#将products按价格从高到低排
rank_list = np.arange(11,0,-1)

rank_list1 = np.arange(11,0,-1)
rank1 = np.arange(1,11)
for i in range(1,11):
    rank_list1 = np.vstack((rank_list1,np.insert(rank1,i,11)))
rank_list1 = rank_list1[1:,:]#喜欢便宜的

rank_list2 = np.arange(11,0,-1)
rank2 = np.arange(10,0,-1)
for i in range(1,11):
    rank_list2 = np.vstack((rank_list2,np.insert(rank2,i,11)))
rank_list2 = rank_list2[1:,:]#喜欢便宜的
breakpoint()
rank_list = np.concatenate((rank_list[np.newaxis,:],rank_list1),0)
rank_list = np.concatenate((rank_list,rank_list2),0)
i=0

l = list(range(1,12))
while i<10:
    per = np.random.permutation(l)
    if (not (per == rank_list).all(1).any()) and (not per[0] == 11):
        rank_list = np.vstack((rank_list,per))
        i += 1

def get_random_sum(total, num):
    random_list = []
    random_value = total
    for i in range(num):
        value = random.uniform(0, total)
        random_list.append(value)
    random_list = list(total*np.array(random_list)/sum(random_list))
    random_list[-1] = total-sum(random_list[:-1])
    return random_list


d1 = get_random_sum(0.6, 10)
cus1 = np.array([0.4]+d1+[0]*20)
d2 = get_random_sum(0.6, 10)
cus2 = np.array([0.4]+[0]*10+d2+[0]*10)
d3 = get_random_sum(0.5, 20)
cus3 = np.array([0.5]+d3[:10]+[0]*10+d3[10:])
d4 = get_random_sum(0.5, 20)
cus4 = np.array([0.5]+[0]*10+d4)

cus = np.vstack((cus1,cus2))
cus = np.vstack((cus,cus3))
cus = np.vstack((cus,cus4))

np.save('save/rank_list.npy',rank_list)
np.save('save/cus_type.npy',cus)
#torch.multinomial(torch.from_numpy(cus),1)

#生成transaction数据
###########################################################################
import cvxpy as cp
prods = list(range(10))#0到9
rl = list(range(31))
MNL_para = cp.Variable(40)#每一类顾客对10个商品的utility
LL=cp.Constant(0)
for type_ in range(4):
    print('start sampling for type '+str(type_))
    lists_dist = cus[type_]
    for person in range(10000):
        ass_onehot = np.zeros(10)
        ass = random.sample(prods,4)
        ass_onehot[ass] = 1
        ass_ind = np.hstack((ass_onehot,np.array([1])))
        ass_ind = (ass_ind*np.arange(1,12)).astype(int).reshape(11,1)
        rl_index = np.random.choice(rl,1,p=lists_dist)[0]
        #选rank list里面出现在了ass中的，排得最靠前的商品
        choose = rank_list[rl_index][np.min(np.where(rank_list[rl_index] == ass_ind)[-1])]
        if choose == 11:#没买
            temp = [0]
            for i in ass:
                temp += [MNL_para[i+type_*10]]
            LL += -cp.log_sum_exp(cp.vstack(temp))
        else:
            temp1=[0]
            temp2=[0]
            temp1 += [MNL_para[choose-1+type_*10]]
            for i in ass:
                temp2 += [MNL_para[i+type_*10]]
            LL += cp.sum(cp.vstack(temp1))-cp.log_sum_exp(cp.vstack(temp2))
objective=cp.Maximize(LL)
constraints=[]
prob=cp.Problem(objective,constraints)
prob.solve(solver='ECOS',verbose=True)

beta = list(list(prob.solution.primal_vars.values())[0])
breakpoint()





















