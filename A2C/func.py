import numpy as np
import torch
from torch.autograd import Variable
import logging

def to_numpy(var, gpu_used=False):
    return var.cpu().data.numpy().astype(np.float64) if gpu_used else var.data.numpy().astype(np.float64)

def to_tensor(ndarray, requires_grad=True, gpu_used=False, gpu_0 = 0):
    if gpu_used:
        return Variable(torch.from_numpy(ndarray).cuda(device=gpu_0).type(torch.cuda.DoubleTensor),
                        requires_grad=requires_grad)#volatile=volatile,
    else:
        return Variable(torch.from_numpy(ndarray).type(torch.DoubleTensor),
                        requires_grad=requires_grad)#volatile=volatile,


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def prob_of_products(ass,V):
    V = ass * V
    V = np.append(V, 1)
    prob = V/np.sum(V)
    return prob#不在assortment里的商品概率是0

def get_myopic_ass(prices,V,inventory_level):
    N = len(inventory_level)
    large_to_small = np.argsort(-prices)  # revenue-ordered结论
    ass_matrix = np.zeros((N, N))
    for i in range(N):
        ass_matrix[i, large_to_small[0:i + 1]] = 1
    inv = inventory_level.copy()
    inv[inv > 0] = 1
    ass_matrix = ass_matrix * inv
    V = ass_matrix * V
    no_click = np.ones((N, 1))
    V = np.concatenate((V, no_click), axis=1)
    prob = V / np.sum(V, axis=1).reshape((-1, 1))
    revenue = prob[:,:-1] @ prices
    return ass_matrix[np.argmax(revenue)]

'''def get_opt_ass(prices,V,inventory_level):
    N = len(inventory_level)
    ass_matrix = np.zeros((N, N))
    V = ass_matrix * V
    no_click = np.ones((N, 1))
    V = np.concatenate((V, no_click), axis=1)
    prob = V / np.sum(V, axis=1).reshape((-1, 1))
    revenue = prob[:,:-1] @ prices
    return ass_matrix[np.argmax(revenue)]'''

def Cardinality_ass(guess_V,profits,constraint):
    intersections = []
    pairs = []
    pairs.append((0, 0))
    intersections.append(-999999999)
    # O(N^2)
    for i in range(len(profits)):
        for j in range(i + 1, len(profits)+1):
            # here our count indexing starts from 0, when it should be 1
            pairs.append((i, j))
            if i == 0:
                intersections.append(profits[j-1])#跟横轴的交点
            else:
                numerator = profits[i-1] * guess_V[i-1] - profits[j-1] * guess_V[j-1]
                denominator = guess_V[i-1] - guess_V[j-1]
                intersections.append(numerator / denominator)
    pairs.append((len(profits), len(profits)))
    intersections.append(999999999)
    args = np.argsort(intersections)
    pairs = np.asarray(pairs)[args]

    A = []
    G = set()
    B = set()
    # v deals with only the inside options, drop the outside option
    sigma = np.argsort(-guess_V)  # descending order
    G.update(sigma[:constraint])
    A.append(sigma[:constraint].tolist())
    for i in range(len(pairs)-1):
        if i == 0:
            continue
        if pairs[i][0] != 0:  # last index(column) will be our 0
            # swap order
            swap_values = pairs[i]-1
            swap_index = np.argwhere(np.isin(sigma, swap_values)).flatten()
            swap_1, swap_2 = sigma[swap_index[0]], sigma[swap_index[1]]
            sigma[swap_index[0]], sigma[swap_index[1]] = swap_2, swap_1
        else:
            B.add(pairs[i][1]-1)
        G = set(sigma[:constraint])
        A_t = G - B
        if A_t:
            A.append(list(A_t))

    profits_ = []
    for assortment in A:
        v = guess_V[assortment]
        w = profits[assortment]
        numerator = np.dot(v, w)
        denominator = 1 + np.sum(v)
        profits_.append(numerator / denominator)

    max_profs_index = np.argmax(profits_)
    ass=A[max_profs_index]
    '''v = V[ass]
    w = profits[ass]
    numerator = np.dot(v, w)
    denominator = 1 + np.sum(v)
    profit=numerator / denominator'''

    ass_onehot=np.array([0]*len(profits))
    ass_onehot[ass] = 1
    return ass_onehot



class MNL:
    '''
    Class to determine the best assortment of items using
    Multinomial Logit Discrete Choice Model
    Algorithm covered in Rusmevichientong et al. 2010
    '''

    def __init__(self, V, guess_V, profits):
        '''
        mean_utility : mean utility (shifted to set outside option to 0),
                        includes outside option, outside option is last column.
        cardinality : maximum number of products that can be presented
        cust_pref vector = (e^(mu_i)) for i = {1,...,N}
                         = 1 for i =0
        profit vector = 0 for i = 0
                      = profit for others
        '''
        #self.utility = np.asarray(mean_utility)
        #self.cust_pref = np.exp(self.utility)
        self.cust_pref = V
        self.true_cust_pref = guess_V
        self.profits = profits

        assert (self.profits.shape == self.cust_pref.shape)

    def find_intersections(self):
        '''
        finds all the intersection points and sorts them in ascending order
        I(i_t, j_t) are the x coordinates of intersection points to sort
        lambda = I(i,j) = \frac{v_iw_i - v_jw_j}{v_i - v_j}
        to enumerate A(lambda) for all lambda - it is sufficient to enumerate all intersections point of lines
        returns array of all zipped (i,j) and intersection pair
        '''
        intersections = []
        pairs = []

        pairs.append((0, 0))
        intersections.append(-999999999)
        # O(N^2)
        for i in range(len(self.profits)):
            for j in range(i + 1, len(self.profits)+1):
                # here our count indexing starts from 0, when it should be 1
                pairs.append((i, j))
                if i == 0:
                    intersections.append(self.profits[j-1])#跟横轴的交点
                else:
                    numerator = self.profits[i-1] * self.cust_pref[i-1] - self.profits[j-1] * self.cust_pref[j-1]
                    denominator = self.cust_pref[i-1] - self.cust_pref[j-1]
                    intersections.append(numerator / denominator)
        #inter_pairs = np.asarray(list(zip(pairs, intersections)))
        #index_sorting_intersections = np.argsort(inter_pairs[:, 1])
        #inter_pairs = inter_pairs[index_sorting_intersections]
        # add the 2 end points, (0,0) and (K+1, K+1)
        pairs.append((len(self.profits), len(self.profits)))
        intersections.append(999999999)

        args = np.argsort(intersections)
        pairs = np.asarray(pairs)[args]
        #intersections = np.asarray(intersections)[args]
        return pairs

    def staticMNL(self, pairs, constraint):
        '''
        performs staticMNL algorithm, returns collection of assortments
        recall that iterating through intersections is sufficient for all lambda
        \sigma^0 = sorted v in descending order
        for intersection:
            update sigma - transpose i and j for I(i,j)
            update(new) G - top C
            update B - if i==0, add j
            update A - G-B
        return A (outside option is value 0, everything else should +1)
        input:
            intersections: sorted intersections :: list of [(i,j), I(i,j)] \forall interactions
            constraint: constraint for number of items in assortment
        '''
        # initialization
        A = []
        G = set()
        B = set()
        # v deals with only the inside options, drop the outside option
        sigma = np.argsort(-self.cust_pref)  # descending order

        G.update(sigma[:constraint])
        A.append(sigma[:constraint].tolist())
        for i in range(len(pairs)-1):
            if i == 0:
                continue
            if pairs[i][0] != 0:  # last index(column) will be our 0
                # swap order
                swap_values = pairs[i]-1
                swap_index = np.argwhere(np.isin(sigma, swap_values)).flatten()
                swap_1, swap_2 = sigma[swap_index[0]], sigma[swap_index[1]]
                sigma[swap_index[0]], sigma[swap_index[1]] = swap_2, swap_1
            else:
                B.add(pairs[i][1]-1)
            G = set(sigma[:constraint])
            A_t = G - B
            if A_t:
                A.append(list(A_t))
        return A

    def best_ass(self, assortments):
        '''
        tabulate profits for each optimal assortment
        [ [(assortment1), profit_assortment1], [(assortment2),profit_assorment2] ......]
        f(s) = \frac{\sum_{j \in S} w_jv_j}{1 + \sum_{j \in S} v_j}
        where s represents the items in assortment
        assortments does not contain the 0 indexed outside option
        input:
            assortments: list of all optimal assortments
        '''
        profits_ = []
        for assortment in assortments:
            v = self.cust_pref[assortment]
            w = self.profits[assortment]
            numerator = np.dot(v, w)
            denominator = 1 + np.sum(v)
            profits_.append(numerator / denominator)

        #assort_profits = list(zip(map(tuple, assortments), profits_))
        max_profs_index = np.argmax(profits_)

        ass=assortments[max_profs_index]
        v = self.true_cust_pref[ass]
        w = self.profits[ass]
        numerator = np.dot(v, w)
        denominator = 1 + np.sum(v)
        profit=numerator / denominator

        return assortments[max_profs_index],profit


