import cvxpy as cp
import numpy as np
import random
import pandas as pd

def estimate_MNL_beta(train_data,X):
    number_samples = len(train_data[0])
    # fit beta on data
    beta = [cp.Variable(5),cp.Variable(5),cp.Variable(5),cp.Variable(5)]
    LL = cp.Constant(0)
    for id_ in range(number_samples):  # 对每一个search id
        arriving_customer = train_data[0][id_]
        ass = train_data[1][id_]
        choice = train_data[2][id_]
        if choice in ass:
            # purchase prob
            temp1 = [0]
            temp2 = [0]
            temp1 += [beta[arriving_customer] @ X[choice]]
            for i in ass:
                temp2 += [beta[arriving_customer] @ X[i]]
            LL += cp.sum(cp.vstack(temp1)) - cp.log_sum_exp(cp.vstack(temp2))
        else:
            # no purchase prob
            temp = [0]
            for i in ass:
                temp += [beta[arriving_customer] @ X[i]]
            LL += -cp.log_sum_exp(cp.vstack(temp))
    objective = cp.Maximize(LL)
    constraints = []
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='ECOS', verbose=True)
    return np.array(list(prob.solution.primal_vars.values()))

def MNL_out_of_sample_log_likelihood(test_data, beta, X):
    number_samples = len(test_data[0])
    LL = 0
    CE = []
    for id_ in range(number_samples):  # 对每一个search id
        arriving_customer = test_data[0][id_]
        ass = test_data[1][id_]
        choice = test_data[2][id_]
        if choice in ass:#买了
            # purchase prob
            temp2 = 1
            temp1 = np.dot(beta[arriving_customer],X[choice])
            for i in ass:
                temp2 += np.exp(np.dot(beta[arriving_customer],X[i]))
            CE.append(-temp1 + np.log(temp2))
            LL += temp1 - np.log(temp2)
        else:
            # no purchase prob
            temp = 1
            for i in ass:
                temp += np.exp(np.dot(beta[arriving_customer],X[i]))
            CE.append(np.log(temp))
            LL += -np.log(temp)
    return LL,np.mean(CE)

'''def create_data(args,seg_prob):
    number_customer_types = args.num_cus_types
    number_products = args.num_products
    number_samples = args.number_samples
    np.random.seed(0)
    random.seed(0)
    number_products_features = 5
    number_customer_features = 3
    X = np.random.rand(number_products, number_products_features)  # 商品的属性
    Z = np.random.rand(number_customer_types, number_customer_features)  # 顾客的属性
    coef = []
    for z in Z:
        coef.append([z[0] * z[1], -z[1] ** 2, -z[0] * z[2],
                     z[1] * z[2], z[2] ** 2])
    coef = np.array(coef) + np.random.rand(4, 5) / 10
    mu = np.mean(coef, axis=0)
    sigma = np.std(coef, axis=0)
    coef = (coef - mu) / sigma  # ground truth系数

    cus_data = []
    ass_data = []
    Y_data = []
    for sam in range(number_samples):
        cus = np.random.choice(range(number_customer_types),p=seg_prob)#到来的customer type
        cus_coef = coef[cus]
        ass = random.sample(range(number_products),3)#货架上的商品
        weights = []
        for prod in ass:
            weights.append(np.exp(np.dot(X[prod],cus_coef)))#+X[prod][0]*X[prod][1]*cus_coef[0]+X[prod][3]*X[prod][4]*cus_coef[0]+(X[prod][3]**2)*cus_coef[0]
        weights.append(1)
        weights = weights/np.sum(weights)
        ass.append(10)
        Y = np.random.choice(ass,p=weights)#选择的商品,10表示没买
        cus_data.append(cus)
        ass_data.append(ass)
        Y_data.append(Y)
    data = [np.array(cus_data),np.array(ass_data)[:,:-1],np.array(Y_data)]
    breakpoint()
    return X,Z,data'''

def create_data(args):
    X_0 = np.load(args.p_file)
    X = X_0[:-1]
    Z = np.eye(4)
    prod_dup = np.repeat(X.reshape(1,-1),4,axis = 0).reshape(30*4,-1)
    prod_dup0 = np.repeat(X_0.reshape(1, -1), 4, axis=0).reshape(31*4, -1)
    cus_dup = np.repeat(Z,30,axis = 0)
    cus_dup0 = np.repeat(Z,31,axis=0)
    return X,Z,np.concatenate((prod_dup,cus_dup),axis=1).reshape(4,30,-1),\
           np.concatenate((prod_dup0,cus_dup0),axis=1).reshape(4,31,-1)


def train_test_split(data):
    number_samples = len(data[0])
    random.seed(0)
    id_list = list(range(number_samples))
    random.shuffle(id_list)
    train_data = [data[0][id_list[0: int(number_samples * 0.6)]],
                  data[1][id_list[0: int(number_samples * 0.6)]],
                  data[2][id_list[0: int(number_samples * 0.6)]]]
    valid_data = [data[0][id_list[int(number_samples * 0.6): int(number_samples * 0.8)]],
                  data[1][id_list[int(number_samples * 0.6): int(number_samples * 0.8)]],
                  data[2][id_list[int(number_samples * 0.6): int(number_samples * 0.8)]]]
    test_data =  [data[0][id_list[int(number_samples * 0.8):]],
                  data[1][id_list[int(number_samples * 0.8):]],
                  data[2][id_list[int(number_samples * 0.8):]]]
    return train_data, valid_data, test_data


def generate_set(args,T,seed_,seg_prob):
    np.random.seed(seed_)
    test_set = np.zeros((args.test_episode,T+20), dtype=int)
    for i in range(args.test_episode):
        input_sequence = np.random.choice \
                (a=np.arange(4), size=T+20, replace=True, p=seg_prob)
        test_set[i] = input_sequence
    val_set = np.zeros((args.val_episode,T+20), dtype=int)
    for i in range(args.val_episode):
        input_sequence = np.random.choice \
                (a=np.arange(4), size=T+20, replace=True, p=seg_prob)
        val_set[i] = input_sequence
    return val_set,test_set

def data_transform(data,X,Z):
    Y = []
    Prod = [[],[],[]]
    Cus_feature = []
    for i in range(len(data[0])):
        cus = data[0][i]
        cus_feature = Cus_feature.append(Z[cus])
        ass = data[1][i]
        if data[2][i] in ass:
            Y.append(np.where(ass == data[2][i])[0][0])
        else:
            Y.append(3)
        for j,p in enumerate(ass):
            Prod[j].append(X[p])
    Cus_feature = np.array(Cus_feature)
    for i in range(3):
        Prod[i] = np.array(Prod[i])
    Prod.append(np.zeros(Prod[0].shape))
    Prod.append(Cus_feature)
    Y = np.array(Y)
    return Prod,Y


def compute_returns(next_value, rewards, m_dones, gamma=1):
    R = next_value
    returns = []
    for step_ in reversed(range(rewards.shape[1])):
        R = rewards[:,step_].unsqueeze(dim=1) + \
            gamma * R * m_dones[step_]
        returns.insert(0, R)
    return returns#t时刻，t+1时刻......对当前expected value的预测

import sys
import os
import logging
_streams = {
    "stdout": sys.stdout
}
def setup_logger(name: str, level: int, stream: str = "stdout") -> logging.Logger:
    global _streams
    if stream not in _streams:
        log_folder = os.path.dirname(stream)
        _streams[stream] = open(stream, 'w')
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    for stream in _streams:
        sh = logging.StreamHandler(stream=_streams[stream])
        sh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger




    
    
    
    
    