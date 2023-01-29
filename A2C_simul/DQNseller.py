import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from func import prob_of_products,Cardinality_ass
torch.set_default_tensor_type(torch.DoubleTensor)
import torch.optim as optim
from uti import create_data
def init_weights(layer):
    if type(layer) == nn.Linear:
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)
        
class QModel(nn.Module):
    def __init__(self, args):
        super(QModel,self).__init__()
        torch.manual_seed(args.net_seed)
        
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
        
        self.num_products = args.num_products
        self.args = args

    def forward(self, x):
        p_e = self.product_encoder(torch.cat((x[:,:self.num_products],x[:,self.num_products+self.args.num_cus_types:]), dim=1))#
        #c_e = self.cus_encoder(x[:,self.num_products:self.num_products+self.args.num_cus_types])#
        c_e = x[:,self.num_products:self.num_products+self.args.num_cus_types]
        x = self.share(torch.cat((p_e, c_e), dim=1))
        #计算每一个商品的Q值，最后一项是不选择这个行为的Q值，等于0
        #x_no_click = torch.zeros((x.shape[0], 1)).to(self.args.device)
        return x#torch.cat([x, x_no_click], dim=1)


class DQN_Seller:
    def __init__(self,args,MNL_para):
        self.args = args
        self.device = args.device
        self.MNL_para = MNL_para
        X, Z, concat_feature,_ = create_data(args)
        self.products_feature = concat_feature
        self.cardinality=args.cardinality
        self.N = args.num_products
        self.num_cus_types = args.num_cus_types
        self.cus_type = np.eye(args.num_cus_types)
        
        self.total_reward=0
        self.replayMemory = deque()
        self.replay_size=0
        self.trainStep = 0
        self.global_step = 0
        # init Q network
        self.Q_Net = QModel(args).to(args.device)
        # init Target network
        self.QT_Net = QModel(args).to(args.device)
        self.copyTargetQNetwork()
        self.type_='train'
        self.epsilon=0.9
        self.epsilonEnd=0.1
        self.batchSize = 32
        # 定义optimizer
        self.optimizer = torch.optim.Adam(self.Q_Net.parameters(), lr=args.share_lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer,step_size=args.step,gamma=args.lr_decay_lambda)
        self.loss_func = nn.MSELoss()  # 默认reduction=mean
    def set_type(self,type_):
        self.type_=type_
    def reset(self):
        self.global_step=0
        self.total_reward = 0
    def assortment(self,state,inventory_level,batchSize,mask,arriving_seg):
        Q_s_i=self.Q_Net.forward(torch.from_numpy(state).double().to(self.device)).cpu().squeeze().detach().numpy()##
        Q_s_i[mask] = 0
        if self.type_=='train':#epsilon-greedy去选择action
            if (random.random()<=self.epsilon) or (self.replay_size < 500):
                ass = np.zeros((batchSize,self.N))
                for i in range(batchSize):
                    random_choose = random.sample(range(self.N), self.cardinality)
                    ass[i][random_choose] = 1
                ass = ass*inventory_level
                ass[ass>0] = 1
            else:
                SQ_ass = []
                for i,cus in enumerate(arriving_seg):
                    V = np.exp((self.products_feature[cus] @ self.MNL_para).ravel())
                    SQ_ass.append(
                        Cardinality_ass(V,Q_s_i[i],self.cardinality))
                ass = np.array(SQ_ass)
                ass = ass*inventory_level
                ass[ass>0] = 1
                
        else:#test的时候就只用网络选择action
            SQ_ass = []
            for i,cus in enumerate(arriving_seg):
                V = np.exp((self.products_feature[cus] @ self.MNL_para).ravel())
                SQ_ass.append(
                    Cardinality_ass(V,Q_s_i[i],self.cardinality))
            ass = np.array(SQ_ass)
            ass = ass*inventory_level
            ass[ass>0] = 1
        return ass


    def learn(self,state,ass,reward,nextObservation,terminal):
        self.replayMemory.append([state,ass,reward,nextObservation,terminal])#将记录加入到replay buffer中去
        self.replay_size = len(self.replayMemory)
        if self.replay_size > 1000000:  # replayMemory最长不能超过100000
            self.replayMemory.popleft()  # replayMemory是个双向队列，replaySize已经超过上限了，就从左边踢掉一个
            self.train_QNet()
            state_ = 'train'
            self.trainStep += 1
        elif self.replay_size >= 5000:  # 开始训练的replayMemory最短5000
            # Train the network
            self.train_QNet()
            state_ = 'train'
            self.trainStep += 1
        else:
            state_='ob'
        if terminal and state_ == "train":
            self.epsilonReduce()
        self.global_step += 1

    def epsilonReduce(self):
        if self.epsilon > self.epsilonEnd:
            self.epsilon -= 0.8/800

    def train_QNet(self):
        # step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, self.batchSize)  # 从replayMemory中sample大小为64的样本出来
        state_batch = torch.from_numpy(np.array([data[0] for data in minibatch])).double()
        ass_batch = torch.from_numpy(np.array([data[1] for data in minibatch])).double()
        reward_batch = torch.from_numpy(np.array([data[2] for data in minibatch])).double().to(self.device)  # dim: [batchSize个]
        next_state_batch = np.array([data[3] for data in minibatch])
        Terminal = torch.tensor([1-data[4] for data in minibatch]).to(self.device)
        # step2: calculate Q Value and y
        with torch.no_grad():
            #根据next_state_batch选择对应的next_assortment_batch，进而得到每一个next_state对应的 Q*(s,a),即 V*(s),就是对应的 optimal value
            QTValue_batch = self.QT_Net.forward(torch.from_numpy(next_state_batch).double().to(self.device))
            QTValue_batch = torch.cat([QTValue_batch, torch.zeros((self.batchSize, 1)).to(self.device)], dim=1)
            
            cus_type = next_state_batch[:, self.N:self.N+self.num_cus_types]
            arriving_seg = np.nonzero(cus_type)[1].reshape(self.batchSize,1)
            inventory = next_state_batch[:, :self.N]
            mask = inventory.copy()
            mask[inventory == 0] = 1
            mask[inventory > 0]=0
            
            self.type_=='update'
            target_assortment = self.assortment(next_state_batch,inventory,self.batchSize,torch.from_numpy(mask).bool(),arriving_seg)
            self.type_=='train'
            next_ass_batch = torch.from_numpy(target_assortment).double()
            
            p_feature_batch = self.products_feature[arriving_seg][:,0,:,:]
            V = torch.from_numpy(np.exp(p_feature_batch @ self.MNL_para))
            V = next_ass_batch*V
            no_click = torch.ones((self.batchSize, 1))
            V = torch.cat([V, no_click], dim=1)
            prob_tensor = (V/torch.sum(V,dim=1).reshape((-1,1))).double().to(self.device)
            next_R_batch=torch.sum(QTValue_batch * prob_tensor, dim=1)
        y_batch = reward_batch.ravel() + Terminal * next_R_batch # target Q Network算出来的Q值

        QValue_batch = self.Q_Net.forward(state_batch.to(self.device))
        QValue_batch = torch.cat([QValue_batch, torch.zeros((self.batchSize, 1)).to(self.device)], dim=1)
        cus_type = next_state_batch[:, self.N:self.N+self.num_cus_types]
        arriving_seg = np.nonzero(cus_type)[1].reshape(self.batchSize,1)
        p_feature_batch = self.products_feature[arriving_seg][:,0,:,:]
        V = torch.from_numpy(np.exp(p_feature_batch @ self.MNL_para))
        V = next_ass_batch*V
        no_click = torch.ones((self.batchSize, 1))
        V = torch.cat([V, no_click], dim=1)
        prob_tensor = (V/torch.sum(V,dim=1).reshape((-1,1))).double().to(self.device)
        R_batch = torch.sum(QValue_batch * prob_tensor, dim=1)#Q Network算出来的Q值
        # step3:梯度下降
        loss = self.loss_func(R_batch, y_batch)#希望Q Network和target Q Network接近
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.optimizer.param_groups[0]['lr']>self.args.lr_min:
            self.lr_scheduler.step()
        # step4:间隔100就将当前QNetwork copy给Target QNetwork
        if self.trainStep % 50 == 1:
            #print('训练次数',self.trainStep)
            self.copyTargetQNetwork()
    def copyTargetQNetwork(self):
        self.QT_Net.load_state_dict(self.Q_Net.state_dict())
    
    def save_model(self, args):
        torch.save(
            self.Q_Net.state_dict(),
            'save/BestQNet'+args.num+'.pt'
        )
        self.args.logger.info('Qmodel weights saved')

    def load_weights(self ,args):
        self.Q_Net.load_state_dict(
            torch.load('save/BestQNet'+args.num+'.pt')
        )
        print('Qmodel weights loaded')