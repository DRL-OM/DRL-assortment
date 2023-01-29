from net import A2C
from DQNseller import DQN_Seller
from func import *
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)
from env import market_dynamic
from other_agents import OA_agent,myopic_agent,E_IB_agent
import torch,math
from numpy import *
from uti import compute_returns
import time

def clip_grad_norms(args,param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"],
            max_norm
            if max_norm > 0
            else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2,
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    )
    if args.print_grad:
        print(grad_norms, max_norm)
    return grad_norms, grad_norms_clipped

def plot_box(args,name,OA_list,myopic_list,E_IB_list,seller_list,plot=True):
    data = [np.array(seller_list), np.array(OA_list),
                np.array(myopic_list), np.array(E_IB_list)]
    import matplotlib.pyplot as plt
    labels = ['A2C', 'Random', 'Myopic', 'EIB']
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    # rectangular box plot
    bplot1 = ax1.boxplot(data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    # ax1.set_title('Rectangular box plot')
    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen', 'orchid']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    # adding horizontal grid lines
    ax1.yaxis.grid(True)
    plt.tick_params(labelsize=22)
    fig.savefig(r'plot/'+name+'.pdf',
                dpi=600, format='pdf')

def initialize(args,model):
    optimizer = optim.Adam(
        [#{"params": model.inv_encoder.parameters()},
         #{"params": model.price_encoder.parameters()},
         {"params": model.product_encoder.parameters()},
         #{"params": model.cus_encoder.parameters()},
         {"params": model.share.parameters()},
         {"params": model.actor.parameters(), "lr": args.actor_lr},
         {"params": model.critic.parameters(), "lr": args.critic_lr}],
          lr=args.share_lr
    )#model.actor.state_dict()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.step,gamma=args.lr_decay_lambda
    )
    return lr_scheduler,optimizer

def train_batch(args,
    model, lr_scheduler, optimizer, env, input_sequence):
    # Evaluate model, get costs and log probabilities
    i = 0
    total_reward = 0
    while not (env.all_finished()):#隔几步更新一次
        log_probs, values, rewards, mean_entropy, m_dones, i, next_value = \
            model.roll_out(env, input_sequence, i)
        total_reward += rewards.sum(1).mean()
        returns = compute_returns(next_value, rewards, m_dones)#包括了部分真实（实施action之后的）的回报，而values全是虚假的
        returns = torch.cat(returns,1).detach()#batch_size*args.num_steps
        advantage = returns - values#大于0表示action是好的
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = args.a_rate*actor_loss + args.c_rate * critic_loss \
               - args.e_rate * mean_entropy#30,20,20
        # Perform backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        # Clip gradient norms and get (clipped) gradient norms for logging
        #grad_norms = clip_grad_norms(args, optimizer.param_groups, args.max_norm)#change inplace
        #breakpoint()
        optimizer.step()
        #breakpoint()
        if optimizer.param_groups[0]['lr']>args.lr_min:
            lr_scheduler.step()
    return total_reward

def Q_batch(args, DQN_seller, env, input_sequence):
    # Evaluate model, get costs and log probabilities
    total_reward = 0
    i = 0
    arriving_seg = input_sequence[:,i]
    cus_type = DQN_seller.cus_type[arriving_seg]
    s = np.hstack((env.inventory_level/env.initial_inventory, cus_type))
    if args.use_price:
        s = np.hstack((s,np.tile(env.products_price, (env.batch_size, 1))))
    while True:#隔几步更新一次
        mask = torch.from_numpy(env.get_mask())
        #action
        assortment = DQN_seller.assortment(s,env.inventory_level,env.batch_size,mask.bool(),arriving_seg)
        #feedback
        _, reward = env.step(arriving_seg,assortment)
        if (env.all_finished()):
            break
        i += 1
        arriving_seg = input_sequence[:,i]
        cus_type = DQN_seller.cus_type[arriving_seg]
        next_s = np.hstack((env.inventory_level/env.initial_inventory, cus_type))
        if args.use_price:
            next_s = np.hstack((next_s,np.tile(env.products_price, (env.batch_size, 1))))
        
        if DQN_seller.type_=='train':
            for ii in range(env.batch_size):
                DQN_seller.learn(s[ii],assortment[ii],reward[ii],next_s[ii],False)
        total_reward += reward.sum(1).mean()
        s = next_s
    if DQN_seller.type_=='train':
        for ii in range(env.batch_size):
            DQN_seller.learn(s[ii],assortment[ii],reward[ii],s[ii],True)
    return total_reward

def val(args,T,env,DQNenv,seller,DQN_seller,initial_inventory,val_set):
    env.batch_size = args.batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.batch_size, 1))
    DQNenv.batch_size = args.batch_size
    DQNenv.initial_inventory = np.tile(initial_inventory, (args.batch_size, 1))
    DQNenv.inventory_level = np.tile(initial_inventory, (args.batch_size, 1))
    
    episodes = int(len(val_set) / env.batch_size)
    val_set = np.split(val_set, episodes)
    seller_list = np.zeros((env.batch_size, 1))
    DQN_seller_list = np.zeros((env.batch_size, 1))
    for i in range(episodes):
        #print('validate episode: ', i, ' / ', episodes)
        T_ = T
        if args.change_T:
            T_ = np.random.randint(T - 10, T + 10)
        input_sequence = val_set[i][:,:T_]
        env.reset(initial_inventory, T_)
        DQNenv.reset(initial_inventory, T_)
        if args.A2C:
            cost,test_value = seller.test_env(env, input_sequence)
            seller_list = np.vstack((seller_list, cost))
        DQN_seller.type_ = 'test'
        if args.SQ:
            total_reward_Q = Q_batch(args, DQN_seller, DQNenv, input_sequence)
            DQN_seller_list = np.vstack((DQN_seller_list, total_reward_Q))
        #print(env.inventory_level)
        #breakpoint()
    if args.A2C:    
        seller_list = list(seller_list.ravel()[env.batch_size:])
    if args.SQ:
        DQN_seller_list = list(DQN_seller_list.ravel()[env.batch_size:])
    
    env.batch_size = args.train_batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.train_batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.train_batch_size, 1))
    DQNenv.batch_size = args.train_batch_size
    DQNenv.initial_inventory = np.tile(initial_inventory, (args.train_batch_size, 1))
    DQNenv.inventory_level = np.tile(initial_inventory, (args.train_batch_size, 1))
    DQN_seller.type_ = 'train'
    if args.A2C and args.SQ:
        return mean(seller_list),mean(DQN_seller_list)
    if args.A2C:
        return mean(seller_list)
    if args.SQ:
        return mean(DQN_seller_list)

def train(args,ResNet,seg_prob,products_price,
          initial_inventory,val_set,logger,MNL_para):
    T = args.selling_length#总的时间长度
    env = market_dynamic(args,ResNet,seg_prob,
                         initial_inventory, products_price, T,True)
    input_length = 2*len(initial_inventory)+args.num_cus_types#
    seller = A2C(args, input_length).to(args.device)
    
    DQNenv = market_dynamic(args,ResNet,seg_prob,
                         initial_inventory, products_price, T,True)
    DQN_seller = DQN_Seller(args,MNL_para)
    
    if args.load:
        seller.load_weights(args)
    lr_scheduler,optimizer = initialize(args,seller)
    best_total_reward = 0
    best_total_reward_Q = 0
    T_ = T
    np.random.seed(args.seed)
    for epoch in range(args.epoch_num):
        if epoch%50==0:
            logger.info("start epoch: {} / {}".format(epoch+1,args.epoch_num))
            logger.info("learning rate now: {},{},{}".format(optimizer.param_groups[1]['lr'],optimizer.param_groups[2]['lr'],
                                                         optimizer.param_groups[3]['lr'])) 
        #生成训练的sequence
        if args.change_T:
            T_ = np.random.randint(T-10,T+10)
        env.reset(initial_inventory, T_)
        DQNenv.reset(initial_inventory, T_)
        input_sequence = np.zeros((args.train_batch_size,T_), dtype=int)
        for j in range(args.train_batch_size):
            input_sequence_ = np.random.choice \
                    (a=np.arange(args.num_cus_types), size=T_, replace=True, p=seg_prob)
            input_sequence[j] = input_sequence_
        
        seller.set_decode_type('sampling')
        if args.A2C:
            total_reward = train_batch(args, seller, lr_scheduler, optimizer, env, input_sequence)
        if args.SQ:
            total_reward_Q = Q_batch(args, DQN_seller, DQNenv, input_sequence)
        #logger.info('train reward: {:.4f}'.format(total_reward))
        #print(env.inventory_level)
        #breakpoint()
        seller.set_decode_type('greedy')
        if args.A2C:
            total_val_reward = val(args,T_,env,DQNenv,seller,DQN_seller,initial_inventory,val_set)
            logger.info("mean validate reward: {:.4f}".format(total_val_reward))
            if total_val_reward>best_total_reward:
                seller.save_model(args)
                best_total_reward = total_val_reward
        if args.SQ:
            total_val_reward_Q = val(args,T_,env,DQNenv,seller,DQN_seller,initial_inventory,val_set)
            logger.info("mean validate reward: {:.4f}".format(total_val_reward_Q))
            if total_val_reward_Q>best_total_reward_Q:
                DQN_seller.save_model(args)
                best_total_reward_Q = total_val_reward_Q
            
        
def other_agents(OA_seller,myopic_seller,E_IB_seller,
                 initial_inventory,T,products_price,input_sequence):
    OA_seller.reset(initial_inventory, T)
    myopic_seller.reset(initial_inventory, T, products_price)
    E_IB_seller.reset(initial_inventory, T)
    for t in range(T-2):
        arriving_seg = input_sequence[:, t].reshape(-1, 1)
        OA_seller.step(arriving_seg)
        myopic_seller.step(arriving_seg)
        E_IB_seller.step(arriving_seg)
    #breakpoint()

def test(ResNet,MNL_para,seg_prob,initial_inventory,T,products_price,args,test_set,logger,load=True,plot=True):
    env_OA = market_dynamic(args,ResNet,seg_prob,initial_inventory,products_price,T)
    env_myopic = market_dynamic(args,ResNet,seg_prob,initial_inventory,products_price,T)
    env_EIB = market_dynamic(args,ResNet,seg_prob,initial_inventory,products_price,T)
    OA_seller = OA_agent(args,env_OA, products_price)
    myopic_seller = myopic_agent(args,env_myopic,MNL_para,products_price)
    E_IB_seller = E_IB_agent(args,env_EIB,MNL_para,products_price)
    OA_list = np.zeros((args.batch_size, 1))
    myopic_list = np.zeros((args.batch_size, 1))
    E_IB_list = np.zeros((args.batch_size, 1))

    env = market_dynamic(args,ResNet,seg_prob,initial_inventory,products_price,T)
    input_length = 2*len(initial_inventory)+args.num_cus_types#
    seller = A2C(args, input_length).to(args.device)
    DQNenv = market_dynamic(args,ResNet,seg_prob,
                         initial_inventory, products_price, T,True)
    DQN_seller = DQN_Seller(args,MNL_para)
    if load and (not args.test_benchmark):
        seller.load_weights(args)
        DQN_seller.load_weights(args)
    seller.set_decode_type('greedy')
    DQN_seller.type_ = 'test'
    #seller.set_decode_type('sampling')
    seller_list = np.zeros((args.batch_size, 1))
    DQN_seller_list = np.zeros((env.batch_size, 1))
    episodes = int(len(test_set)/args.batch_size)
    test_set = np.split(test_set,episodes)
    for i in range(episodes):
        #print('test episode: ',i,' / ',episodes)
        T_ = T
        if args.change_T:
            T_ = np.random.randint(T - 10, T + 10)
        env.reset(initial_inventory, T_)
        input_sequence = test_set[i][:,:T_]
        if bool(not plot) and args.num_products==10:# 
            other_agents(OA_seller, myopic_seller, E_IB_seller,
                     initial_inventory, T_, products_price, input_sequence)
        OA_list = np.vstack((OA_list,OA_seller.total_reward))
        myopic_list = np.vstack((myopic_list, myopic_seller.total_reward))
        E_IB_list = np.vstack((E_IB_list, E_IB_seller.total_reward))
        if not args.test_benchmark:
            if args.A2C:
                cost,test_value = seller.test_env(env, input_sequence)
                seller_list = np.vstack((seller_list, cost))
            if args.SQ:
                total_reward_Q = Q_batch(args, DQN_seller, DQNenv, input_sequence)
                DQN_seller_list = np.vstack((DQN_seller_list, total_reward_Q))
            
        else:
            cost = np.zeros((args.batch_size, 1))
        #logger.info("value change: {}".format(test_value))
        #breakpoint()
    OA_list = list(OA_list.ravel()[args.batch_size:])
    myopic_list = list(myopic_list.ravel()[args.batch_size:])
    E_IB_list = list(E_IB_list.ravel()[args.batch_size:])
    if args.A2C:    
        seller_list = list(seller_list.ravel()[args.batch_size:])
    if args.SQ:    
        DQN_seller_list = list(DQN_seller_list.ravel()[args.batch_size:])
        seller_list = DQN_seller_list
    logger.info("mean test reward1: {}".format(seller_list))
    logger.info("mean test reward2: {}".format(OA_list))
    logger.info("mean test reward3: {}".format(myopic_list))
    logger.info("mean test reward4: {}".format(E_IB_list))
    logger.info("mean test reward: {:.4f},{:.4f},{:.4f},{:.4f}".format(mean(seller_list), mean(OA_list),
          mean(myopic_list), mean(E_IB_list)))
    if not plot:
        print(mean(seller_list)/mean(OA_list),mean(OA_list)/mean(OA_list),
              mean(myopic_list)/mean(OA_list),mean(E_IB_list)/mean(OA_list))
    return OA_list,myopic_list,E_IB_list,seller_list





