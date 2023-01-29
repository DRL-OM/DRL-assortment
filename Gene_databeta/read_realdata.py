import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import random_split
from feature import encoder,Res_Assort_Net,train_test_split,estimate_MNL_beta,MNL_out_of_sample_log_likelihood
import matplotlib.pyplot as plt
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('../../expedia/real_data.csv',index_col=0)

num_prods = len(data['prop_id'].unique())
num_prods_features = 6
num_cus_types = 4
'''
prop_features = []#每个hotel的特征
for prop in data['prop_id'].unique():
    prop_features.append(data[data['prop_id']==prop].iloc[0,3:9])
prop_features.append(np.zeros(num_prods_features))
prop_features = np.array(prop_features)#(58, 6)shape
np.save('save/expedia_prop_features.npy',prop_features)

#创建数据集：product feature, customer feature, assortment01
sample_list = np.zeros((1,num_prods+1,num_prods_features+num_cus_types+num_prods+1))
choose_list = []
i=0
for srch in data['srch_id'].unique():
    if i%1000==0:
        print('第：',str(i),'个 search / 9270')
    srch_data = data[data['srch_id'] == srch]
    show_prods = srch_data.loc[:, 'prop_id'].values
    ass = np.zeros(num_prods+1)
    ass[show_prods] = 1
    ass[-1] = 1
    ass = np.repeat(ass[np.newaxis,:],num_prods+1,0)#(58, 58)

    multi = np.zeros((num_prods+1,num_prods_features))
    multi[show_prods] = 1
    prop_fea = prop_features*multi#(58, 6)
    cus_fea = np.repeat(srch_data.iloc[0,-4:].values[np.newaxis,:],num_prods+1,0)#(58, 4)
    sample = np.concatenate((np.concatenate((prop_fea,cus_fea),1),ass),1)#(58, 68)
    if srch_data['booking_bool'].sum() == 0:
        choose = 57
    else:
        choose = srch_data[srch_data['booking_bool'] == 1]['prop_id'].values[0]
    sample_list = np.concatenate((sample_list,sample[np.newaxis,:,:]),0)
    choose_list.append(choose)
    i+=1
sample_list = sample_list[1:]
breakpoint()
choose_list = np.array(choose_list)
np.save('save/expedia_sample.npy',sample_list)
np.save('save/expedia_choose.npy',choose_list)
'''

sample_list = np.load('save/expedia_sample.npy')
choose_list = np.load('save/expedia_choose.npy')
num_samples = len(choose_list)
sample_list = torch.from_numpy(sample_list)
sample_list = sample_list.float()
choose_list = torch.from_numpy(choose_list)
choose_list = choose_list.reshape(num_samples,1)
choose_list = np.repeat(choose_list,num_prods+1,1).reshape(num_samples,num_prods+1,1)
choose_list = choose_list.type(torch.LongTensor)
sample_list = sample_list.to(device)
choose_list = choose_list.to(device)

dataset = TensorDataset(sample_list,choose_list)
train_data,test_data = random_split(dataset,
                  [round(0.8*num_samples),
                    round(0.2*num_samples)],
                  generator=torch.Generator().manual_seed(0))
batch_size = 32
train_iter = DataLoader(train_data,batch_size=batch_size,
                        shuffle=True,num_workers=0)
test_iter = DataLoader(test_data,batch_size=batch_size,
                       shuffle=True,num_workers=0)

#实例化网络，建立训练模型
product_encoder = encoder(num_prods_features,2,20,20).to(device)
cus_encoder = encoder(num_cus_types,2,20,20).to(device)
net = Res_Assort_Net(116, 1, 116, 3).to(device)
print('cuda:',torch.cuda.is_available())
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(itertools.chain(product_encoder.parameters(),
                              cus_encoder.parameters(),
                              net.parameters()),lr = 0.0001)#定义优化器，设置学习率
epochs = 30#训练轮数

test_runningloss = 0
with torch.no_grad():
    #net.eval()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout
    for ass,choice in test_iter:
        prod = ass[:, :, :num_prods_features]
        cus = ass[:, :, num_prods_features:num_prods_features+num_cus_types]
        ass_onehot = ass[:, 0, num_prods_features+num_cus_types:]  # batch_size*35
        choose = choice[:, 0, 0]
        optimizer.zero_grad()
        e_prod = product_encoder(prod)
        e_cust = cus_encoder(cus)
        latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*58
        y_hat = net(latent_uti, ass_onehot)
        test_runningloss += lossFunc(y_hat,choose)
print("初始测试误差：",str(test_runningloss/len(test_iter)))     

train_loss, test_loss = [], []
print("开始训练Res-Assort-Net")
for e in range(epochs):
    running_loss = 0
    for ass,choice in train_iter:
        prod = ass[:, :, :num_prods_features]
        cus = ass[:, :, num_prods_features:num_prods_features+num_cus_types]
        ass_onehot = ass[:,0,num_prods_features+num_cus_types:]#batch_size*58
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
    with torch.no_grad():
        #net.eval()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout
        for ass,choice in test_iter:
            prod = ass[:, :, :num_prods_features]
            cus = ass[:, :, num_prods_features:num_prods_features+num_cus_types]
            ass_onehot = ass[:, 0, num_prods_features+num_cus_types:]  # batch_size*35
            choose = choice[:, 0, 0]
            optimizer.zero_grad()
            e_prod = product_encoder(prod)
            e_cust = cus_encoder(cus)
            latent_uti = torch.sum(e_prod * e_cust, dim=2)  # batch_size*58
            y_hat = net(latent_uti, ass_onehot)
            test_runningloss += lossFunc(y_hat,choose)
    #net.train()#不写这两个程序也可以运行，这是因为这两个方法是针对在网络训练和测试时采用不同方式的情况，比如Batch Normalization 和 Dropout

    train_loss.append(running_loss/len(train_iter))
    test_loss.append(test_runningloss.item()/len(test_iter))
    print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
          "训练误差: {:.3f}.. ".format(running_loss / len(train_iter)),
          "测试误差: {:.3f}.. ".format(test_runningloss / len(test_iter)))

torch.save(product_encoder, 'ex_product_encoder.pth')
torch.save(cus_encoder, 'ex_cus_encoder.pth')
torch.save(net, 'ex_net.pth')

plt.figure()
ax = plt.gca().axes
MNL_loss = [1.492]*epochs
MNL_test_loss = [1.517]*epochs
#Markov_loss = [0.788]*epochs
plt.plot(MNL_test_loss,color='#edae49',label='MNL test loss')
plt.plot(MNL_loss,color='#d1495b',label='MNL train loss')
#plt.plot(Markov_loss,label='Markov test loss')
plt.plot(np.array(test_loss),color='#00798c',label='Net test loss')
plt.plot(np.array(train_loss),color='#003d5b',label='Net train loss')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1);####设置左边坐标轴的粗细
ax.tick_params(direction='in', width=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.xlabel("Training Epoch",size=20)
plt.ylabel('Crossentropy Loss',size=20)
#plt.show()
plt.savefig('fig/expedia.pdf',dpi=600,bbox_inches = 'tight')


