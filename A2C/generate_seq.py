import numpy as np
import pandas as pd
T = 220
seg_prob = [0.25]*4
dict_ = {}
for i in range(20000):
    input_sequence = np.random.choice \
            (a=np.arange(4), size=T, replace=True, p=seg_prob)
    dict_[i] = input_sequence

def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....}
    # 列名即为CSV中数据对应的列名， 数据为一个列表
    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储
    # 默认存储在当前路径下
    Name = []
    times = 0
    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))
            times += 1
        Pd_data = pd.DataFrame(columns=Name, data=Data)
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))
            times += 1
        Pd_data = pd.DataFrame(index=Name, data=Data)
    if Save_format == 'csv':
        Pd_data.to_csv('./' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('./' + file_name + '.xls', encoding='utf-8')

Save_to_Csv(dict_,'sequences',Save_format='csv', Save_type='row')