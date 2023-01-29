# Deep Reinforcement Learning for Dynamic Assortment Customization: An End-to-End Framework

The code of the paper `Deep Reinforcement Learning for Dynamic Assortment Customization: An End-to-End Framework` is presented at this repository. The code works with `Python3.9` and `torch1.13`. The folder  `Gene_databeta` is used for data and parameters generating.  The folder  `A2C` is used for semi-synthetic experiment.  The folder  `A2C_simul` is used for synthetic simulation with resnet choice model as environment.  The folder  `A2C_rl` is used for synthetic simulation with rank-list choice model as environment.  

#### Gene_databeta

The file `read_realdata.py` is used for reading from extracted expedia data in the csv form, and transforming it into training data for resnet choice model.

The file `random_data.py` is used for generating feature and resnet model randomly,  and generating transaction data from environment for training MNL model. Results are saved in folder simul. 

The file `rl.py` is used for generating rank-list choice model, and generating transaction data to fit MNL model.

#### A2C_simul

`main.py` is the file to call to start the training. `train.py` is used for training and testing. `net.py` instantiates the A2C agent and `other_agents.py` provides benchmark agents instantiations . `env.py` simulates the rank-list based environment. `feature.py` instantiates the resnet choice model.

to run the training for 10 products, each with 10 initial inventory levels, 4 types of customers, and Load Factor 1.0 :

```python
python main.py --A2C=True --num_products=10 --ini_inv=10 --num_cus_types=4 --selling_length=100 --seed_range=20 --net_seed=10 --seed=0 --share_lr=0.01 --actor_lr=0.001 --critic_lr=0.1 --step=100 --lr_min=0.0001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.99
```

to run with Load Factor 0.6, 0.8 and 1.1, change selling_length to 60, 80 and 110.

to run the training for 100 products, each with 2 initial inventory levels, 4 types of customers, and Load Factor 0.75 :

```python
python main.py --A2C=True --num_products=100 --ini_inv=2 --num_cus_types=4 --selling_length=150 --seed_range=20 --net_seed=10 --seed=0 --share_lr=0.01 --actor_lr=0.001 --critic_lr=0.01 --step=50 --lr_min=0.0001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.999
```

#### A2C_rl

The file `learn_MNL.py` is used for generating transaction data from resnet for fitting MNL model.

to run the training for 10 products, each with 10 initial inventory levels, 4 types of customers, and Load Factor 1.8 :

```python
python main.py --num_products=10 --ini_inv=10 --num_cus_types=4 --selling_length=180 --seed_range=20 --net_seed=0 --seed=0 --share_lr=0.01 --actor_lr=0.001 --critic_lr=0.01 --step=50 --lr_min=0.0001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.99
```

--detail=True : see the change of inventory and critic in the testing process.

#### A2C

run with :

```python
python main.py --num_products=57 --ini_inv=2 --num_cus_types=4 --selling_length=200 --seed_range=20 --net_seed=1 --share_lr=0.0001 --actor_lr=0.0001 --critic_lr=0.01 --step=20 --lr_min=0.00001 --e_rate=0.001 --a_rate=1 --c_rate=1 --lr_decay_lambda=0.99
```

