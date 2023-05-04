from net import A2C
from func import *
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)
from env import market_dynamic
from other_agents import OA_agent,myopic_agent,E_IB_agent,sub_t_agent
import torch,math
from numpy import *
from uti import compute_returns

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

def train(args,ResNet,products_price,initial_inventory,train_sequences,logger):
    torch.manual_seed(args.net_seed)
    random.seed(args.net_seed)
    input_length = 2*len(initial_inventory)+args.num_cus_types#
    seller = A2C(args, input_length).to(args.device)
    lr_scheduler,optimizer = initialize(args,seller)
    if args.load:
        seller.load_weights(args)
    best_total_reward = 0
    
    env = market_dynamic(args,ResNet,initial_inventory, products_price, 100,True)
    
    seller.set_decode_type('greedy')
    total_val_reward = val(args,env,seller,initial_inventory,train_sequences)
    logger.info("initial mean reward: {:.4f}".format(total_val_reward))
    
    for epoch in range(args.epoch_num):
        args.trans_record = np.zeros((1,32))
        logger.info("start epoch: {} / {}".format(epoch+1,args.epoch_num))
        logger.info("learning rate now: {},{},{}".format(optimizer.param_groups[1]['lr'],optimizer.param_groups[2]['lr'],
                                                         optimizer.param_groups[3]['lr']))
        seq_indexlist = np.arange(len(train_sequences))
        np.random.shuffle(seq_indexlist)
        for seq_index in seq_indexlist:
            input_sequence = train_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])-1
            env.reset(initial_inventory, T)
            seller.set_decode_type('sampling')
            total_reward = train_batch(args, seller, lr_scheduler, optimizer, env, input_sequence)
            #logger.info('train reward: {:.4f}'.format(total_reward))
            #print(env.inventory_level)
            #breakpoint()
        seller.set_decode_type('greedy')
        total_val_reward = val(args,env,seller,initial_inventory,train_sequences)
        logger.info("mean validate reward: {:.4f}".format(total_val_reward))
        if total_val_reward>best_total_reward:
            seller.save_model(args)
            best_total_reward = total_val_reward
        np.save(r'MNL/'+ args.name+'/'+'round '+str(epoch)+'.npy',args.trans_record[1:])
            
        

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


def val(args,env,seller,initial_inventory,train_sequences):
    env.batch_size = args.batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.batch_size, 1))
    seller_list = np.zeros((env.batch_size, 1))
    
    seq_indexlist = np.arange(len(train_sequences))
    for seq_index in seq_indexlist:
        input_sequence = train_sequences[seq_index]
        T = len(input_sequence)
        input_sequence = np.array([input_sequence])-1
        env.reset(initial_inventory, T)
        cost,test_value = seller.test_env(env, input_sequence)
        seller_list = np.vstack((seller_list, cost))
        #print(env.inventory_level)
        #breakpoint()
    seller_list = list(seller_list.ravel()[env.batch_size:])
    
    env.batch_size = args.train_batch_size
    env.initial_inventory = np.tile(initial_inventory, (args.train_batch_size, 1))
    env.inventory_level = np.tile(initial_inventory, (args.train_batch_size, 1))
    return mean(seller_list)
        
        
def test(test_sequences,ResNet,MNL_para,initial_inventory,products_price,args,logger,load=True,plot=True):
    OA_list_mean=np.zeros(len(test_sequences))
    myopic_list_mean=np.zeros(len(test_sequences))
    E_IB_list_mean=np.zeros(len(test_sequences))
    sub_t_list_mean=np.zeros(len(test_sequences))
    seller_list_mean=np.zeros(len(test_sequences))
    for seed_ in range(args.seed_range):
        logger.info("test seed: {}************************************".format(seed_))
        #np.random.seed(seed_)
        #random.seed(seed_)
        env_OA = market_dynamic(args,ResNet,initial_inventory,products_price,100)
        env_myopic = market_dynamic(args,ResNet,initial_inventory,products_price,100)
        env_EIB = market_dynamic(args,ResNet,initial_inventory,products_price,100)
        env_sub = market_dynamic(args,ResNet,initial_inventory,products_price,100)
        
        OA_seller = OA_agent(args,env_OA, products_price)
        myopic_seller = myopic_agent(args,env_myopic,MNL_para,products_price)
        E_IB_seller = E_IB_agent(args,env_EIB,MNL_para,products_price)
        sub_t_seller = sub_t_agent(args,env_sub,MNL_para,products_price)
        
        OA_list = np.zeros((args.batch_size, 1))
        myopic_list = np.zeros((args.batch_size, 1))
        E_IB_list = np.zeros((args.batch_size, 1))
        sub_t_list = np.zeros((args.batch_size, 1))

        env = market_dynamic(args,ResNet,initial_inventory, products_price, 100,True)
        input_length = 2*len(initial_inventory)+args.num_cus_types#
        seller = A2C(args, input_length).to(args.device)
        if load and (not args.test_benchmark):
            seller.load_weights(args)
        seller.set_decode_type('greedy')
        #seller.set_decode_type('sampling')
        seller_list = np.zeros((args.batch_size, 1))
        
        check = False
        seq_indexlist = np.arange(len(test_sequences))
        for seq_index in seq_indexlist:
            input_sequence = test_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])-1
            env.reset(initial_inventory, T)
            if bool(not plot):# 
                other_agents(OA_seller, myopic_seller, E_IB_seller,sub_t_seller,
                         initial_inventory, T, products_price, input_sequence)
            OA_list = np.vstack((OA_list,OA_seller.total_reward))
            myopic_list = np.vstack((myopic_list, myopic_seller.total_reward))
            E_IB_list = np.vstack((E_IB_list, E_IB_seller.total_reward))
            sub_t_list = np.vstack((sub_t_list, sub_t_seller.total_reward))
            #print(myopic_seller.total_reward,E_IB_seller.total_reward,sub_t_seller.total_reward)
            if not args.test_benchmark:
                cost,test_value = seller.test_env(env, input_sequence)
                seller_list = np.vstack((seller_list, cost))
            else:
                cost = np.zeros((args.batch_size, 1))
        OA_list = list(OA_list.ravel()[args.batch_size:])
        myopic_list = list(myopic_list.ravel()[args.batch_size:])
        E_IB_list = list(E_IB_list.ravel()[args.batch_size:])
        sub_t_list = list(sub_t_list.ravel()[args.batch_size:])
        seller_list = list(seller_list.ravel()[args.batch_size:])
        logger.info("mean test reward1: {}".format(seller_list))
        logger.info("mean test reward2: {}".format(OA_list))
        logger.info("mean test reward3: {}".format(myopic_list))
        logger.info("mean test reward4: {}".format(E_IB_list))
        logger.info("mean test reward4: {}".format(sub_t_list))
        logger.info("mean test reward: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(mean(seller_list), mean(OA_list),
              mean(myopic_list), mean(E_IB_list), mean(sub_t_list)))
        if not plot:
            print(mean(seller_list)/mean(OA_list),mean(OA_list)/mean(OA_list),
                  mean(myopic_list)/mean(OA_list),mean(E_IB_list)/mean(OA_list))
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        sub_t_list_mean = sub_t_list_mean+np.array(sub_t_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
    if not args.test_benchmark:    
        plot_box(args, args.name,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
                 E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))

def other_agents(OA_seller,myopic_seller,E_IB_seller,sub_t_seller,
                 initial_inventory,T,products_price,input_sequence,check = False):
    OA_seller.reset(initial_inventory, T)
    myopic_seller.reset(initial_inventory, T, products_price)
    E_IB_seller.reset(initial_inventory, T)
    sub_t_seller.reset(initial_inventory, T, products_price)
    for t in range(T-1):
        arriving_seg = input_sequence[:, t].reshape(-1, 1)
        OA_seller.step(arriving_seg,check)
        myopic_seller.step(arriving_seg,check)
        E_IB_seller.step(arriving_seg,check)
        sub_t_seller.step(arriving_seg,check)
    #breakpoint()
        
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