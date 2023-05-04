from net import A2C
from func import *
import torch.optim as optim
torch.set_default_tensor_type(torch.DoubleTensor)
from env import market_dynamic,market_dynamic_Net
from other_agents import OA_agent,myopic_agent,E_IB_agent,optmyopic_agent,optEIB_agent,sub_t_agent,LP_agent
import torch,math
from numpy import *
from uti import compute_returns


def train(args,Gated_net,products_price,initial_inventory,train_sequences,logger):
    torch.manual_seed(args.net_seed)
    random.seed(args.net_seed)
    input_length = 2*len(initial_inventory)+args.num_cus_types#
    seller = A2C(args, input_length).to(args.device)
    lr_scheduler,optimizer = initialize(args,seller)
    if args.load:
        seller.load_weights(args)
    best_total_reward = 0
    env = market_dynamic_Net(args,Gated_net,initial_inventory, products_price,True)
    seller.set_decode_type('greedy')
    total_val_reward = val(args,env,seller,initial_inventory,train_sequences)
    logger.info("initial mean reward: {:.4f}".format(total_val_reward))
    
    for epoch in range(args.epoch_num):
        logger.info("start epoch: {} / {}".format(epoch+1,args.epoch_num))
        logger.info("learning rate now: {},{},{}".format(optimizer.param_groups[1]['lr'],optimizer.param_groups[2]['lr'],
                                                         optimizer.param_groups[3]['lr']))
        seq_indexlist = np.arange(len(train_sequences))
        np.random.shuffle(seq_indexlist)
        for seq_index in seq_indexlist:
            input_sequence = train_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])
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
        #print(optimizer.param_groups[-1]['lr'])
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
        input_sequence = np.array([input_sequence])
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


def test(test_sequences,MNL_para,initial_inventory,products_price,args,logger,load=True,plot= False):
    OA_list_mean=np.zeros(len(test_sequences))
    myopic_list_mean=np.zeros(len(test_sequences))
    E_IB_list_mean=np.zeros(len(test_sequences))
    sub_t_list_mean=np.zeros(len(test_sequences))
    LP_list_mean=np.zeros(len(test_sequences))
    seller_list_mean=np.zeros(len(test_sequences))
    for seed_ in range(args.seed_range):
        if args.test_opt:
            env_opt = market_dynamic(args,initial_inventory,products_price)
            opt_agent = optmyopic_agent(args, env_opt, products_price)
            opt_list = []
            env_EIBopt = market_dynamic(args,initial_inventory,products_price)
            EIBopt_agent = optEIB_agent(args, env_EIBopt, products_price)
            EIBopt_list = []
            seq_indexlist = np.arange(len(test_sequences))
            for seq_index in seq_indexlist:
                input_sequence = test_sequences[seq_index]
                T = len(input_sequence)
                input_sequence = np.array([input_sequence])
                opt_agent.reset(initial_inventory, T, products_price)
                EIBopt_agent.reset(initial_inventory, T)
                for t in range(T-1):
                    arriving_seg = input_sequence[:, t].reshape(-1, 1)
                    opt_agent.step(arriving_seg)
                    EIBopt_agent.step(arriving_seg)
                opt_list.append(opt_agent.total_reward)
                EIBopt_list.append(EIBopt_agent.total_reward)
                logger.info("opt,EIBopt mean test reward: {}, {}".format(opt_agent.total_reward,EIBopt_agent.total_reward))
            logger.info("******opt mean test reward******: {}".format(np.mean(opt_list)))
            logger.info("******EIBopt mean test reward******: {}".format(np.mean(EIBopt_list)))
            continue
        
        env_OA = market_dynamic(args,initial_inventory,products_price)
        env_myopic = market_dynamic(args,initial_inventory,products_price)
        env_EIB = market_dynamic(args,initial_inventory,products_price)
        env_sub = market_dynamic(args,initial_inventory,products_price)
        env_LP = market_dynamic(args,initial_inventory,products_price)
        
        OA_seller = OA_agent(args,env_OA, products_price)
        myopic_seller = myopic_agent(args,env_myopic,MNL_para,products_price)
        E_IB_seller = E_IB_agent(args,env_EIB,MNL_para,products_price)
        sub_t_seller = sub_t_agent(args,env_sub,MNL_para,products_price)
        LP_seller = LP_agent(args,env_LP,MNL_para,products_price)
        
        OA_list = np.zeros((args.batch_size, 1))
        myopic_list = np.zeros((args.batch_size, 1))
        E_IB_list = np.zeros((args.batch_size, 1))
        sub_t_list = np.zeros((args.batch_size, 1))
        LP_list = np.zeros((args.batch_size, 1))
    
        if args.detail:
            change_of_invOA_list=[]
            change_of_ROA_list=[]
            change_of_invM_list=[]
            change_of_RM_list=[]
            change_of_invE_list=[]
            change_of_RE_list=[]
            change_of_invS_list=[]
            change_of_RS_list=[]
            test_value_list=[]
            change_of_inv_list=[]
            change_of_R_list=[]

        env = market_dynamic(args,initial_inventory,products_price)
        input_length = 2*len(initial_inventory)+args.num_cus_types#
        seller = A2C(args, input_length).to(args.device)
        if load:
            seller.load_weights(args)
        seller.set_decode_type('greedy')
        #seller.set_decode_type('sampling')
        seller_list = np.zeros((args.batch_size, 1))
        
        seq_indexlist = np.arange(len(test_sequences))
        
        check = False
        for seq_index in seq_indexlist:
            input_sequence = test_sequences[seq_index]
            T = len(input_sequence)
            input_sequence = np.array([input_sequence])
            env.reset(initial_inventory, T)
            if not plot:
                if args.detail:
                    change_of_invOA,change_of_ROA,change_of_invM,change_of_RM,change_of_invE,change_of_RE,change_of_invS,change_of_RS = other_agents(OA_seller, myopic_seller, E_IB_seller,sub_t_seller,LP_seller,initial_inventory, T, products_price, input_sequence,args,logger,args.detail,check)
                else:
                    other_agents(OA_seller, myopic_seller, E_IB_seller,sub_t_seller,LP_seller,
                         initial_inventory, T, products_price, input_sequence,args,logger,args.detail,check)
                    check = False
            OA_list = np.vstack((OA_list,OA_seller.total_reward))
            myopic_list = np.vstack((myopic_list, myopic_seller.total_reward))
            E_IB_list = np.vstack((E_IB_list, E_IB_seller.total_reward))
            sub_t_list = np.vstack((sub_t_list, sub_t_seller.total_reward))
            LP_list = np.vstack((LP_list, LP_seller.total_reward))
            
            #print(myopic_seller.total_reward,E_IB_seller.total_reward,sub_t_seller.total_reward,LP_seller.total_reward)
            #breakpoint()
            if args.detail:
                cost,test_value,change_of_inv,change_of_R = seller.test_env(env, input_sequence)
                change_of_invOA_list.append(np.array(change_of_invOA))
                change_of_ROA_list.append(np.array(change_of_ROA))
                change_of_invM_list.append(np.array(change_of_invM))
                change_of_RM_list.append(np.array(change_of_RM))
                change_of_invE_list.append(np.array(change_of_invE))
                change_of_RE_list.append(np.array(change_of_RE))
                change_of_invS_list.append(np.array(change_of_invS))
                change_of_RS_list.append(np.array(change_of_RS))
                test_value_list.append(np.array(test_value))
                change_of_inv_list.append(np.array(change_of_inv))
                change_of_R_list.append(np.array(change_of_R))
            else:
                cost,test_value = seller.test_env(env, input_sequence)
            #logger.info("value change: {}".format(test_value))
            #breakpoint()
            #cost = np.zeros((args.batch_size, 1))
            seller_list = np.vstack((seller_list, cost))
        if args.detail:
            np.save('change/'+args.name+'change_of_invOA_list'+str(seed_)+'.npy',np.array(change_of_invOA_list))
            np.save('change/'+args.name+'change_of_ROA_list'+str(seed_)+'.npy',np.array(change_of_ROA_list))
            np.save('change/'+args.name+'change_of_invM_list'+str(seed_)+'.npy',np.array(change_of_invM_list))
            np.save('change/'+args.name+'change_of_RM_list'+str(seed_)+'.npy',np.array(change_of_RM_list))
            np.save('change/'+args.name+'change_of_invE_list'+str(seed_)+'.npy',np.array(change_of_invE_list))
            np.save('change/'+args.name+'change_of_RE_list'+str(seed_)+'.npy',np.array(change_of_RE_list))
            np.save('change/'+args.name+'change_of_invS_list'+str(seed_)+'.npy',np.array(change_of_invS_list))
            np.save('change/'+args.name+'change_of_RS_list'+str(seed_)+'.npy',np.array(change_of_RS_list))
            np.save('change/'+args.name+'test_value_list'+str(seed_)+'.npy',np.array(test_value_list))
            np.save('change/'+args.name+'change_of_inv_list'+str(seed_)+'.npy',np.array(change_of_inv_list))
            np.save('change/'+args.name+'change_of_R_list'+str(seed_)+'.npy',np.array(change_of_R_list))
        OA_list = list(OA_list.ravel()[args.batch_size:])
        myopic_list = list(myopic_list.ravel()[args.batch_size:])
        E_IB_list = list(E_IB_list.ravel()[args.batch_size:])
        sub_t_list = list(sub_t_list.ravel()[args.batch_size:])
        LP_list = list(LP_list.ravel()[args.batch_size:])
        seller_list = list(seller_list.ravel()[args.batch_size:])
        logger.info("mean test reward1: {}".format(seller_list))
        logger.info("mean test reward2: {}".format(OA_list))
        logger.info("mean test reward3: {}".format(myopic_list))
        logger.info("mean test reward4: {}".format(E_IB_list))
        logger.info("mean test reward4: {}".format(sub_t_list))
        logger.info("mean test reward4: {}".format(LP_list))
        logger.info("mean test reward: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(mean(seller_list), mean(OA_list),
              mean(myopic_list), mean(E_IB_list), mean(sub_t_list), mean(LP_list)))
        if not plot:
            print(mean(seller_list)/mean(OA_list),mean(OA_list)/mean(OA_list),
                  mean(myopic_list)/mean(OA_list),mean(E_IB_list)/mean(OA_list))
        OA_list_mean = OA_list_mean+np.array(OA_list)
        myopic_list_mean = myopic_list_mean+np.array(myopic_list)
        E_IB_list_mean = E_IB_list_mean+np.array(E_IB_list)
        sub_t_list_mean = sub_t_list_mean+np.array(sub_t_list)
        LP_list_mean = sub_t_list_mean+np.array(LP_list)
        seller_list_mean = seller_list_mean+np.array(seller_list)
    plot_box(args,OA_list_mean/(args.seed_range), myopic_list_mean/(args.seed_range),
                 E_IB_list_mean/(args.seed_range), seller_list_mean/(args.seed_range))


def other_agents(OA_seller,myopic_seller,E_IB_seller,sub_t_seller,LP_seller,
                 initial_inventory,T,products_price,input_sequence,args,logger,detail=False,check = False):
    OA_seller.reset(initial_inventory, T)
    myopic_seller.reset(initial_inventory, T, products_price)
    E_IB_seller.reset(initial_inventory, T)
    sub_t_seller.reset(initial_inventory, T, products_price)
    LP_seller.reset(initial_inventory, T, products_price)
    change_of_ROA = []
    change_of_invOA = []
    change_of_RM = []
    change_of_invM = []
    change_of_RE = []
    change_of_invE = []
    change_of_RS = []
    change_of_invS = []
    for t in range(T-1):
        arriving_seg = input_sequence[:, t].reshape(-1, 1)
        OA_seller.step(arriving_seg,check)
        myopic_seller.step(arriving_seg,check)
        E_IB_seller.step(arriving_seg,check)
        sub_t_seller.step(arriving_seg,check)
        #LP_seller.step(arriving_seg,check)
        #print(E_IB_seller.total_reward,sub_t_seller.total_reward,LP_seller.total_reward)
        check = False
        if detail:
            logger.info("inventory_level of OA_seller: {}".format(OA_seller.market.inventory_level.mean(0)))
            logger.info("inventory_level of myopic_seller: {}".format(myopic_seller.market.inventory_level.mean(0)))
            logger.info("inventory_level of E_IB_seller: {}".format(E_IB_seller.market.inventory_level.mean(0)))
            logger.info("inventory_level of E_IB_seller: {}".format(sub_t_seller.market.inventory_level.mean(0)))
            logger.info("other agents' reward: {} , {} , {}, {}".
                        format(OA_seller.total_reward.mean(0),myopic_seller.total_reward.mean(0)
                               ,E_IB_seller.total_reward.mean(0),sub_t_seller.total_reward.mean(0))
                       )
            change_of_ROA.append(OA_seller.total_reward.mean(0)[0])
            change_of_invOA.append(OA_seller.market.inventory_level.mean(0))
            change_of_RM.append(myopic_seller.total_reward.mean(0)[0])
            change_of_invM.append(myopic_seller.market.inventory_level.mean(0))
            change_of_RE.append(E_IB_seller.total_reward.mean(0)[0])
            change_of_invE.append(E_IB_seller.market.inventory_level.mean(0))
            change_of_RS.append(sub_t_seller.total_reward.mean(0)[0])
            change_of_invS.append(sub_t_seller.market.inventory_level.mean(0))
    if detail:
        return change_of_invOA,change_of_ROA,change_of_invM,change_of_RM,change_of_invE,change_of_RE,change_of_invS,change_of_RS
        np.save('change/'+args.name+'change_of_invOA.npy',np.array(change_of_invOA))
        np.save('change/'+args.name+'change_of_ROA.npy',np.array(change_of_ROA))
        np.save('change/'+args.name+'change_of_invM.npy',np.array(change_of_invM))
        np.save('change/'+args.name+'change_of_RM.npy',np.array(change_of_RM))
        np.save('change/'+args.name+'change_of_invE.npy',np.array(change_of_invE))
        np.save('change/'+args.name+'change_of_RE.npy',np.array(change_of_RE))
        np.save('change/'+args.name+'change_of_invS.npy',np.array(change_of_invS))
        np.save('change/'+args.name+'change_of_RS.npy',np.array(change_of_RS))


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

def plot_box(args,OA_list,myopic_list,E_IB_list,seller_list,plot=True):
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
    fig.savefig(r'plot/'+args.name+'.pdf',
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


