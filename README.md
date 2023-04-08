# Deep Reinforcement Learning for Online Assortment Customization: A Data-Driven Approach

The code of the paper `Deep Reinforcement Learning for Online Assortment Customization: A Data-Driven Approach` is presented at this repository. The code works with `Python3.9` and `torch1.13`. The folder  `rlGT` is used for simulation with rank-list ground truth.  The folder  `rlGT_reuse` is used for simulation with reusable products and rank-list ground truth. The folder  `expedia` is used for expedia data pre-processing.  The folder  `realdata` is used for simulation with expedia data.  

#### rlGT

`main.py` is the file to call to start the training.  `uti.py` includes functions for data generating and choice model fitting, and the results are saved in folder `GT`. `train.py` is used for training and testing. `net.py` instantiates the A2C agent and `other_agents.py` provides benchmark agents instantiations . `env.py` simulates the rank-list based environment. `feature.py` instantiates the resnet choice model. 

to run the training for 10 products, each with 10 initial inventory levels, 4 types of customers, cardinality 4, and Load Factor 1.0 :

```python
python main.py --A2C=True --num_products=10 --ini_inv=10 --num_cus_types=4 --cardinality=4 --seed_range=20 --net_seed=10 --net_seed=0 --share_lr=0.001 --actor_lr=0.0001 --critic_lr=0.0001 --step=100 --lr_min=0.00001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.999
```

to run with different Load Factors, change ini_inv to 8, 9, 11.

--detail=True : see the change of inventory and critic in the testing process.

#### rlGT_reuse

different usage time distribution is chosen by line 71~74 and line 153~156 in `env.py` .

run with :

```python
python main.py --A2C=True --num_products=10 --ini_inv=10 --num_cus_types=4 --cardinality=4 --seed_range=20 --net_seed=10 --net_seed=0 --share_lr=0.001 --actor_lr=0.0001 --critic_lr=0.0001 --step=100 --lr_min=0.0001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.999
```

#### realdata

 `read_realdata.py` fits a gated-Assort-Net from expedia data to act as the environment feedback.

run with :

```python
python main.py --num_products=30 --ini_inv=2 --num_cus_types=4 --selling_length=200 --seed_range=20 --net_seed=47 --share_lr=0.005 --actor_lr=0.005 --critic_lr=0.01 --step=50 --lr_min=0.00001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.999
```

