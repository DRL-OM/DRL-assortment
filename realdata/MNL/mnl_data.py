import numpy as np
import pandas as pd
import os
import warnings
 
warnings.filterwarnings('ignore')

prop_features = np.load('../resnet/expedia_prop_features.npy')
prop_features = prop_features[:-1]
name = 'A2C2023-02-18-17-42-55'
path = name
file_list = os.listdir(path)

colname = ['srch','f1','f2','f3','f4','f5','f6','buy']
df1 = [pd.DataFrame(columns=colname)]*20
df2 = [pd.DataFrame(columns=colname)]*20
df3 = [pd.DataFrame(columns=colname)]*20
df4 = [pd.DataFrame(columns=colname)]*20
num = [0,0,0,0]
i=0
for file in file_list:
    print('round '+str(i))
    if not file[-3:]=='npy':
        continue
    trans_data = np.load(path+'/'+file)
    
    for trans in trans_data:#顾客类别（取值0,1,2,3其中一个），assortment one hot 30维，不包括不选，最后的选项0到30，30代表不选
        type_ = int(trans[0])
        ass = trans[1:-1].astype(int)
        prop_fea = prop_features[ass.nonzero()[0]]
        choose = trans[-1]
        if choose != 30:
            ass[int(choose)] = 2
        choice = (ass[ass.nonzero()[0]]-1).reshape((len(prop_fea),1))
        impression = np.hstack(( np.array([num[type_]]*len(prop_fea)).reshape((len(prop_fea),1)) , np.hstack(( prop_fea , choice ))
                        ))
        num[type_] += 1
        if type_ == 0:
            df_i = pd.DataFrame(impression, columns=colname)
            df1[i] = df1[i].append(df_i)
        elif type_ == 1:
            df_i = pd.DataFrame(impression, columns=colname)
            df2[i] = df2[i].append(df_i)
        elif type_ == 2:
            df_i = pd.DataFrame(impression, columns=colname)
            df3[i] = df3[i].append(df_i)
        else:
            df_i = pd.DataFrame(impression, columns=colname)
            df4[i] = df4[i].append(df_i)
    i+=1
    if i>10:
        break
if not os.path.exists(path+'/mat'):
    os.mkdir(path+'/mat')


df1= pd.concat(df1)
df2= pd.concat(df2)
df3= pd.concat(df3)
df4= pd.concat(df4)
    
import scipy.io as scio
dat_ = np.array(df1)
Ind = list(df1.index)
Col = list(df1.columns)
scio.savemat(path+'/mat'+'/type1'+'.mat',{'data':dat_,'index':Ind,'cols':Col})
dat_ = np.array(df2)
Ind = list(df2.index)
Col = list(df2.columns)
scio.savemat(path+'/mat'+'/type2'+'.mat',{'data':dat_,'index':Ind,'cols':Col})
dat_ = np.array(df3)
Ind = list(df3.index)
Col = list(df3.columns)
scio.savemat(path+'/mat'+'/type3'+'.mat',{'data':dat_,'index':Ind,'cols':Col})
dat_ = np.array(df4)
Ind = list(df4.index)
Col = list(df4.columns)
scio.savemat(path+'/mat'+'/type4'+'.mat',{'data':dat_,'index':Ind,'cols':Col})
    

    
    