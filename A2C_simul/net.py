import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
import time
from uti import create_data

def init_weights(layer):
    if type(layer) == nn.Linear:
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)
def init_weights_c(layer):
    if type(layer) == nn.Linear:
        nn.init.constant_(layer.weight, init_c)
        nn.init.constant_(layer.bias, init_c)

def E_penalty_function(x):
    return (1-np.exp(-x))*(np.e/(np.e-1))

class A2C(nn.Module):
    def __init__(self, args,n_states):
        super(A2C, self).__init__()
        torch.manual_seed(args.net_seed)
        # action network definition
        '''self.inv_encoder = nn.Sequential(
            nn.Linear(args.num_products, args.num_products),
            nn.ReLU()
        )
        self.price_encoder = nn.Sequential(
            nn.Linear(args.num_products, args.num_products),
            nn.ReLU()
        )
        self.cus_encoder = nn.Sequential(
            nn.Linear(args.num_cus_types, args.num_cus_types),#
            nn.ReLU()
        )'''
        self.product_encoder = nn.Sequential(
            nn.Linear(2*args.num_products, args.num_products),
            nn.ReLU()
        )
        share_width = args.num_products+args.num_cus_types#
        #share_width = 2*args.num_products+args.num_cus_features
        share = []
        share.append(nn.Linear(share_width, args.w[0]))
        share.append(nn.ReLU())
        for h in range(args.h-1):
            share.append(nn.Linear(args.w[h], args.w[h+1]))
            share.append(nn.ReLU())
        share.append(nn.Linear(args.w[args.h-1], args.nn_out))
        share.append(nn.ReLU())
        self.share = nn.Sequential(*share)
        self.share.apply(init_weights)
        self.critic = nn.Sequential(
            nn.Linear(args.nn_out, 1)
        )
        global init_c
        init_c = 2*args.selling_length/args.nn_out
        self.critic.apply(init_weights_c)
        self.actor = nn.Sequential(
            nn.Linear(args.nn_out, args.num_products)
        )
        self.actor.apply(init_weights)
        self.decode_type = None
        self.num_products = args.num_products
        self.cardinality = args.cardinality
        self.args = args
        self.device = args.device
        self.total_T = args.selling_length
        self.est_T = args.selling_length
        self.cus_type = np.eye(args.num_cus_types)
        X, Z, _,concat_feature0 = create_data(args)
        self.cus_fea = Z
        
    def forward(self, x):
        p_e = self.product_encoder(torch.cat((x[:,:self.num_products],x[:,self.num_products+self.args.num_cus_types:]), dim=1))#
        #c_e = self.cus_encoder(x[:,self.num_products:self.num_products+self.args.num_cus_types])#
        c_e = x[:,self.num_products:self.num_products+self.args.num_cus_types]
        x = self.share(torch.cat((p_e, c_e), dim=1))
        #x = self.share(x)
        value = self.critic(x)
        score = self.actor(x)
        return score, value
    def roll_out(self, env, input_sequence, i):
        ass_log_softmax_list_ = []
        values = []
        R = []
        m_dones=[]
        mean_entropy = 0
        s = 0
        for num_step in range(self.args.num_steps):
            arriving_seg = input_sequence[:,i]
            cus_type = self.cus_type[arriving_seg]#
            s = np.hstack((env.inventory_level/env.initial_inventory, cus_type))#/env.initial_inventory
            #r_ = E_penalty_function(env.inventory_level / env.initial_inventory) * env.products_price
            if self.args.use_price:
                s = np.hstack((s,
                        np.tile(env.products_price, (env.batch_size, 1))))
            #else:
                #s = np.hstack((s, r_))
            # 关键语句
            pre = time.time()
            score, value = self.forward(torch.from_numpy(s).double().to(self.device))
            now = time.time()
            score = score.cpu()
            value = value.cpu()
            #print('time:', now-pre)
            #breakpoint()
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            assortment, entropy, ass_log_softmax = self._select_node(env,
                score, mask.bool())  # Squeeze out steps dimension
            mean_entropy += entropy
            _, reward = env.step(arriving_seg,assortment.numpy())
            #要输出的东西
            ass_log_softmax_list_.append(ass_log_softmax)
            values.append(value)
            m_dones.append(1 - env.all_finished())
            R.append(torch.DoubleTensor(reward))
            i += 1
            if num_step == self.args.num_steps-1 or env.all_finished():#这个roll out结束
                if env.all_finished():
                    next_value = 0#done之后这个值是没用的
                    break
                if num_step == self.args.num_steps-1:
                    next_state = np.hstack((env.inventory_level/env.initial_inventory,self.cus_type[input_sequence[:,i+1]]))#
                    if self.args.use_price:
                        next_state = np.hstack((next_state,
                                           np.tile(env.products_price, (env.batch_size, 1))))
                    _, next_value = self.forward(torch.DoubleTensor(next_state).to(self.device))
                    next_value = value.cpu()
                break
        # Collected lists, return Tensor
        return (
            torch.stack(ass_log_softmax_list_, 1),#shape(batch_size,T)
            torch.cat(values,1),
            torch.cat(R,1),###
            torch.DoubleTensor(mean_entropy),
            m_dones,
            i,
            next_value
        )

    def calc_entropy(self, _log_p):
        entropy = -(_log_p * _log_p.exp()).sum(2).sum(1).mean()
        return entropy

    def _select_node(self, env, score, mask):
        score[mask] = -1e20
        p = torch.log_softmax(score, dim=1)
        dist = Categorical(p)
        entropy = dist.entropy().mean()#想让 entropy 变大，更加随机
        ass = torch.zeros([env.batch_size, self.num_products],dtype=torch.int)
        if self.decode_type == "greedy":
            _, idx1 = torch.sort(p, descending=True)  # descending为False，升序，为True，降序
            selected = idx1[:,:self.cardinality]
            ass.scatter_(1,selected,1)
            ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆
            ass_log_softmax = (ass*p).sum(axis=1)#是一个长度为batch_size的张量
        elif self.decode_type == "sampling":
            selected = p.exp().multinomial(self.cardinality,replacement=True)#有放回的抽样
            #breakpoint()
            ass.scatter_(1,selected,1)
            ass = ass*torch.logical_not(mask)#注意：不是mask着的东西才能摆
            ass_log_softmax = (ass*p).sum(axis=1)
        else:
            assert False, "Unknown decode type"
        return ass, entropy ,ass_log_softmax

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type
    
    def save_model(self, args):
        torch.save(
            self.state_dict(),
            'save/BestNet'+args.num+'.pt'
        )
        self.args.logger.info('model weights saved')

    def load_weights(self ,args):
        self.load_state_dict(
            torch.load('save/BestNet'+args.num+'.pt')
        )
        print('model weights loaded')

    def test_env(self,env,input_sequence):
        R = torch.zeros([env.batch_size, 1])
        i = 0
        test_value = []
        while not (env.all_finished()):
            arriving_seg = input_sequence[:, i]
            cus_type = self.cus_type[arriving_seg]#
            s = np.hstack((env.inventory_level/env.initial_inventory, cus_type))
            #r_ = E_penalty_function(env.inventory_level / env.initial_inventory) * env.products_price
            if self.args.use_price:
                s = np.hstack((s,np.tile(env.products_price, (env.batch_size, 1))))
            #else:
                #s = np.hstack((s, r_))
            # 关键语句
            pre = time.time()
            score, value = self.forward(torch.from_numpy(s).double().to(self.device))
            #breakpoint()
            now = time.time()
            score = score.cpu()
            value = value.cpu()
            test_value.append(np.mean(value.detach().numpy()))
            #print('time:', now-pre)
            #########
            mask = torch.from_numpy(env.get_mask())
            # Select the indices of the next nodes in the sequences
            assortment, p, ass_log_softmax = self._select_node(env,
                            score, mask.bool())  # Squeeze out steps dimension
            _, reward = env.step(arriving_seg, assortment.numpy())
            R += reward
            i += 1
        return R,test_value




