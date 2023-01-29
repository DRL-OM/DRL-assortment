import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import random_split
from feature import encoder,Res_Assort_Net
import matplotlib.pyplot as plt
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X = np.load(r'simul/simul_X.npy')
choose = np.load(r'simul/simul_Y.npy')
num_samples = len(choose)
choose = np.repeat(choose,11,axis=0).reshape(num_samples,11,-1)

cat_data = torch.from_numpy(X)
cat_data = cat_data.float()
choose = torch.from_numpy(choose)
choose = choose.type(torch.LongTensor)
cat_data = cat_data.to(device)
choose = choose.to(device)

dataset = TensorDataset(cat_data,choose)
train_data,test_data = random_split(dataset,
                  [round(0.8*num_samples),
                    round(0.2*num_samples)],
                  generator=torch.Generator().manual_seed(0))
batch_size = 16
train_iter = DataLoader(train_data,batch_size=batch_size,
                        shuffle=True,num_workers=0)
test_iter = DataLoader(test_data,batch_size=batch_size,
                       shuffle=True,num_workers=0)

#实例化网络，建立训练模型
product_encoder = encoder(8,2,40,20).to(device)
cus_encoder = encoder(6,2,40,20).to(device)
net = Res_Assort_Net(22, 1, 22, 2).to(device)
print('cuda:',torch.cuda.is_available())
#net = Gate_Assort_Net(11, 2, 60)
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(itertools.chain(product_encoder.parameters(),
                              cus_encoder.parameters(),
                              net.parameters()),lr = 0.0001)#定义优化器，设置学习率
epochs = 30#训练轮数

train_loss, test_loss = [], []
'''
test_runningloss = 0
with torch.no_grad():
    # net.eval()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout
    for ass, choice in test_iter:
        prod = ass[:, :, 6:14]
        cus = ass[:, :, :6]
        ass_onehot = ass[:, 0, 14:]  # batch_size*35
        choose = choice[:, 0, 0]
        optimizer.zero_grad()
        e_prod = product_encoder(prod)
        e_cust = cus_encoder(cus)
        latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*35
        y_hat = net(latent_uti, ass_onehot)
        test_runningloss += lossFunc(y_hat, choose)
test_loss.append(test_runningloss.item()/len(test_iter))
print("初始测试误差: {:.3f}.. ".format(test_runningloss / len(test_iter)))
'''
print("开始训练Res-Assort-Net")
for e in range(epochs):
    running_loss = 0
    for ass,choice in train_iter:
        prod = ass[:,:,6:14]
        cus = ass[:,:,:6]
        ass_onehot = ass[:,0,14:]#batch_size*35
        choose = choice[:,0,0]
        optimizer.zero_grad()
        e_prod = product_encoder(prod)
        e_cust = cus_encoder(cus)
        latent_uti = torch.sum(e_prod*e_cust,dim=2)#batch_size*35
        y_hat = net(latent_uti,ass_onehot)
        loss = lossFunc(y_hat,choose)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()## 将每轮的loss求和
    test_runningloss = 0
    test_acc = 0
    with torch.no_grad():
        #net.eval()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout
        for ass,choice in test_iter:
            prod = ass[:, :, 6:14]
            cus = ass[:, :, :6]
            ass_onehot = ass[:, 0, 14:] # batch_size*35
            choose = choice[:, 0, 0]
            optimizer.zero_grad()
            e_prod = product_encoder(prod)
            e_cust = cus_encoder(cus)
            latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*35
            y_hat = net(latent_uti, ass_onehot)
            test_runningloss += lossFunc(y_hat,choose)
    #net.train()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout

    train_loss.append(running_loss/len(train_iter))
    test_loss.append(test_runningloss.item()/len(test_iter))
    print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
          "训练误差: {:.3f}.. ".format(running_loss / len(train_iter)),
          "测试误差: {:.3f}.. ".format(test_runningloss / len(test_iter)))

torch.save(product_encoder, 'product_encoder_simul1.pth')
torch.save(cus_encoder, 'cus_encoder_simul1.pth')
torch.save(net, 'net_simul1.pth')
MNL_loss = [0.896]*epochs
#Markov_loss = [0.788]*epochs
plt.plot(MNL_loss,label='MNL test loss')
#plt.plot(Markov_loss,label='Markov test loss')
plt.plot(np.array(train_loss),label='Net train loss')
plt.plot(np.array(test_loss),label='Net test loss')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=16)
#plt.show()
plt.savefig(r'fig/simul_loss.pdf',dpi=600)