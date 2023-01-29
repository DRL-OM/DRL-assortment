import argparse

def init_parser(alg):

    if alg == 'Reinforce':

        parser = argparse.ArgumentParser(description='Thesis')
        parser.add_argument('--logger')
        parser.add_argument('--name')
        parser.add_argument('--gpu', default='0', type=str)
        parser.add_argument('--device')

        parser.add_argument('--gamma', type=float, default=1, metavar='G', help='discount factor for rewards (default: 0.99)')
        parser.add_argument('--epoch_num', type=int, default=3000)
        parser.add_argument('--test_episode', type=int, default=50)
        parser.add_argument('--val_episode', type=int, default=50)
        parser.add_argument('--test_size', type=int, default=0.1, help='')
        parser.add_argument('--val_size', type=int, default=0.5, help='')
        parser.add_argument('--batch_size', default=50, type=int, help='')
        parser.add_argument('--train_batch_size', default=50, type=int, help='')
        parser.add_argument('--num_steps', default=10, type=int, help='')
        parser.add_argument('--h', default=1, type=int, help='hidden layer')
        parser.add_argument('--w', nargs='+', default=[120, 50], type=int)
        parser.add_argument('--nn_out', default=40, type=int)
        parser.add_argument('--share_lr', type=float, default=0.01, help='learning rate.')
        parser.add_argument('--actor_lr', type=float, default=0.001, help='learning rate.')
        parser.add_argument('--critic_lr', type=float, default=0.01, help='learning rate.')
        parser.add_argument('--step', type=float, default=100, help='learning rate decay step.')
        parser.add_argument('--lr_min', type=float, default=1e-4, help='learning rate minimum.')
        parser.add_argument('--e_rate', type=float, default=0.001, help='.')
        parser.add_argument('--a_rate', type=float, default=1, help='.')
        parser.add_argument('--c_rate', type=float, default=1, help='.')
        parser.add_argument('--lr_decay_lambda', type=float, default=0.99, help='.')#0.999999999

        parser.add_argument('--duse_T', type=bool, default=True, help='')
        parser.add_argument('--est_T', type=bool, default=False, help='')
        parser.add_argument('--use_pref', type=bool, default=False, help='')
        parser.add_argument('--same_price', type=bool, default=False, help='')
        parser.add_argument('--use_price', type=bool, default=True, help='')
        parser.add_argument('--only_test', type=bool, default=False, help='')
        parser.add_argument('--change_T', type=bool, default=True, help='')

        parser.add_argument('--num', default='0', type=str, help='number of experiment')
        parser.add_argument('--print_grad', type=bool, default=False, help='')
        parser.add_argument('--max_norm', type=int, default=10, help='')
        parser.add_argument('--seed_range', default=20, type=int, help='')
        parser.add_argument('--net_seed', default=0, type=int, help='')
        parser.add_argument('--seed', default=0, type=int, help='')
        parser.add_argument('--info', type=bool, default=True, help='')
        parser.add_argument('--selling_length', default=180, type=int, help='the length of selling season')
        parser.add_argument('--cardinality', default=4, type=int, help='size constraint')
        parser.add_argument('--ini_inv', default=10, type=int, help='initial inventory')
        parser.add_argument('--num_products', default=10, type=int, help='')
        parser.add_argument('--number_samples', default=10000, type=int, help='')
        parser.add_argument('--num_cus_types', default=4, type=int, help='')
        parser.add_argument('--seg_prob')
        #parser.add_argument('--num_cus_features', default=6, type=int, help='')

        parser.add_argument('--cus_type')
        parser.add_argument('--rank_list')
        parser.add_argument('--detail', type=bool, default=False)
        return parser

    else:

        raise RuntimeError('undefined algorithm {}'.format(alg))