import cvxpy as cp
import numpy as np
import random
import json
from feature import resBlock,Res_Assort_Net,Gate_Assort_Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

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

def create_data():
    return 1,1,1,1


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









def read_json(file_name):
    with open(file_name, 'r') as f:
        data = json.loads(f.read())
    ranked_lists = data['ground_model']['ranked_lists']
    for i in range(1,len(ranked_lists)):
        ranked_lists[i] = ranked_lists[i][:-1]
        ranked_lists[i].insert(random.randint(0,10), 11)
    ranked_lists = np.array(ranked_lists[1:])
    np.save('GT/ranked_lists.npy',ranked_lists)
    
def gene_custype(num_lists=500):
    cus_types = []
    for i in range(4):
        lists_prob = np.random.exponential(1, size=num_lists)
        lists_prob = lists_prob/np.sum(lists_prob)
        cus_types.append(lists_prob)
    return np.array(cus_types)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
def gene_data(ranked_lists,cus_types,cusseg_prob):
    dict_ = {}
    for i in range(200):#sequence data
        T = np.random.randint(90,110)
        input_sequence = []
        for t in range(T):
            arriving_seg = np.random.choice(a=np.arange(4), p=cusseg_prob)
            input_sequence.append(arriving_seg)
        dict_[i] = input_sequence
    seqdata = json.dumps(dict_,cls=NpEncoder)  
    with open('GT/seqdata.json', 'w') as json_file:
        json_file.write(seqdata)
    #Save_to_Csv(dict_,'sequences',Save_format='csv', Save_type='row')
    custype_transdata = {}
    custype_transdata['0'] = []
    custype_transdata['1'] = []
    custype_transdata['2'] = []
    custype_transdata['3'] = []
    for sequence in list(dict_.values())[:160]:#前160天的transaction data
        for arriving_seg in sequence:
            ass = random.sample(list(range(10)), 4)
            #print(ass)
            ass_onehot = np.zeros(11)
            ass_onehot[ass] = 1
            ass_onehot[-1] = 1
            #print(ass_onehot)
            #选rank list里面出现在了ass中的，排得最靠前的商品
            ass_ind = (ass_onehot*np.arange(1,12)).astype(int).reshape(11,1)
            #print(ass_ind)
            rl_index = np.random.choice(list(range(100)),1,p=cus_types[arriving_seg])[0]
            choose = ranked_lists[rl_index][np.min(np.where(ranked_lists[rl_index] == ass_ind)[-1])]
            #print(ranked_lists[rl_index])
            #print(choose)
            custype_transdata[str(arriving_seg)].append(np.append(ass_onehot,(choose-1)))#assortment onehot 表示,以及选的index
    transdata = json.dumps(custype_transdata,cls=NpEncoder)  
    with open('GT/transdata.json', 'w') as json_file:
        json_file.write(transdata)
            

def fit(transdata):
    beta_list = []
    for i in range(4):
        data = transdata[str(i)]
        beta = fitMNL(data)
        beta_list.append(beta)
        fitResNet(data,i)
    beta_list = np.array(beta_list)
    np.save('GT/Learned_MNL_weight.npy',beta_list)

def fitMNL(data):
    MNL_para = cp.Variable(10)#每一类顾客对10个商品的utility
    LL=cp.Constant(0)
    for trans in data:#一个交易记录，前十一位是assortment one hot，最后一位是选择的product index
        ass_onehot = trans[:-2]
        ass = np.flatnonzero(ass_onehot)
        choose = trans[-1]
        if choose == 10:#没选
            temp = [0]
            for i in ass:
                temp += [MNL_para[int(i)]]
            LL += -cp.log_sum_exp(cp.vstack(temp))
        else:
            temp1=[0]
            temp2=[0]
            temp1 += [MNL_para[int(choose)]]
            for i in ass:
                temp2 += [MNL_para[int(i)]]
            LL += cp.sum(cp.vstack(temp1))-cp.log_sum_exp(cp.vstack(temp2))
    objective=cp.Maximize(LL)
    constraints=[]
    prob=cp.Problem(objective,constraints)
    prob.solve(solver='ECOS',verbose=True)
    beta = list(list(prob.solution.primal_vars.values())[0])  
    print(beta)
    LL = 0
    for trans in data:
        ass_onehot = trans[:-2]
        ass = np.flatnonzero(ass_onehot)
        choose = trans[-1]
        if choose == 10:#没选
            temp = 1
            for i in ass:
                temp += np.exp(beta[int(i)])
            LL += -np.log(temp)
        else:
            temp2 = 1
            temp1 = beta[int(choose)]
            for i in ass:
                temp2 += np.exp(beta[int(i)])
            LL += temp1-np.log(temp2)
    print('learned MNL LL:',LL/len(data))
    return beta

def fitResNet(data,type_):
    X = np.array(data)[:,:-1]
    Y = np.array(data)[:,-1]
    #将数据转换成torch形式
    x_train = torch.from_numpy(X)
    x_train = x_train.float()
    y_train = torch.from_numpy(Y)
    y_train = y_train.type(torch.LongTensor)
    batch_size = 16
    datasets_train = TensorDataset(x_train,y_train)
    train_iter = DataLoader(datasets_train,batch_size=batch_size,shuffle=True,num_workers=0)
    net = Gate_Assort_Net(11, 11)
    #net = Res_Assort_Net(11,1,11,2)
    lossFunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr = 0.001)#定义优化器，设置学习率
    epochs = 100#训练轮数
    train_loss = []
    print("开始训练Res-Assort-Net")
    for e in range(epochs):
        running_loss = 0
        for ass,choice in train_iter:
            optimizer.zero_grad()
            y_hat = net(ass)
            loss = lossFunc(y_hat,choice)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()## 将每轮的loss求和
        train_loss.append(running_loss/len(train_iter))
        print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),"训练误差: {:.3f}.. ".format(running_loss / len(train_iter)))
    torch.save(net, 'GT/ResNet'+str(type_)+'.pth')
    
            
if __name__ =='__main__':
    #read_json('GT/top10_rank/30_periods_1_instance.json')
    ranked_lists_all = np.load('GT/ranked_lists.npy')
    #print(len(ranked_lists_all))
    if 0:
        ranked_lists = ranked_lists_all[random.sample(list(range(len(ranked_lists_all))),100)]
        np.save('GT/ranked_lists_use.npy',ranked_lists)
    ranked_lists = np.load('GT/ranked_lists_use.npy')
    #print(ranked_lists)
    if 0:
        cus_types = gene_custype(100)
        np.save('GT/cus_types.npy',cus_types)
    cus_types = np.load('GT/cus_types.npy')
    if 0:
        cusseg_prob = np.random.exponential(1, size=4)
        cusseg_prob = cusseg_prob/np.sum(cusseg_prob)
        np.save('GT/cusseg_prob.npy',cusseg_prob)
    cusseg_prob = np.load('GT/cusseg_prob.npy')
    print(cusseg_prob)
    if 0:
        gene_data(ranked_lists,cus_types,cusseg_prob)
    with open('GT/seqdata.json', 'r') as f:
        seqdata = json.loads(f.read())
    with open('GT/transdata.json', 'r') as f:
        transdata = json.loads(f.read())
    fit(transdata)
    
    
    