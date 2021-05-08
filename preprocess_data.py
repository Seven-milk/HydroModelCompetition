# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import numpy as np
import os

# general set
home = 'H:/文件/水科学数值模拟/初赛/prelim'

# read data
'''假定都是从2017.01.01开始的检验数据
检验数据：给出数据20天，2017.01.01 - 2017.01.20
预报时间：2017.01.21-2017.01.27
'''
train_pre = pd.read_excel(os.path.join(home, 'train', '训练数据集-降雨.xlsx'), index_col=0)
train_runoff = pd.read_excel(os.path.join(home, 'train', '训练数据集-径流.xlsx'), index_col=0)
names = locals()
date_test = pd.date_range(start='2017-01-01 00:00:00', end='2017-01-20 23:00:00', freq='H')
date_predict = pd.date_range(start='2017-01-21 00:00:00', end='2017-01-27 00:00:00', freq='D')
for i in range(5):
    names['test_pre' + str(i)] = pd.read_excel(os.path.join(home, 'test', '检验数据集-降雨.xlsx'), sheet_name=i,
                                               index_col=0, usecols=list(range(1, 22)))
    names['test_runoff' + str(i)] = pd.read_excel(os.path.join(home, 'test', '检验数据集-径流.xlsx'), sheet_name=i,
                                                  index_col=0, usecols=list(range(1, 6)))
    names['test_pre_predict' + str(i)] = pd.read_excel(os.path.join(home, 'test', '检验数据集-预报降雨.xlsx'),
                                                       sheet_name=i, index_col=0, usecols=list(range(21)))
    # set index
    names['test_pre' + str(i)].index = date_test
    names['test_runoff' + str(i)].index = date_test
    names['test_pre_predict' + str(i)].index = date_predict


def preprocess_train():
    # preprocess train data
    print('---------------------------------')
    print("训练数据-降水的缺失值\n", train_pre.isnull().sum())
    print('---------------------------------')
    print("训练数据-径流的缺失值\n", train_runoff.isnull().sum())
    print('---------------------------------')
    print("result: Nan is only found in train_runoff")
    print("preprocess method: average of train_runoff in a day")

    # search Nan index/date in train_runoff
    train_Nan_index = train_runoff[train_runoff.isnull().values == True].index
    train_Nan_date = pd.to_datetime(list(set(train_Nan_index.date)))

    # preprocess train data
    train_pre_D = train_pre.resample("D").sum()  # sum-pre
    train_runoff_D = train_runoff.resample("D").mean()  # mean-runoff

    # save
    train_pre_D.to_excel(os.path.join(home, 'preprocess_data', 'train', 'train_pre_D.xlsx'))
    train_runoff_D.to_excel(os.path.join(home, 'preprocess_data', 'train', 'train_runoff_D.xlsx'))

    np.save(os.path.join(home, 'preprocess_data', 'train', 'train_Nan_index'), train_Nan_index)
    np.save(os.path.join(home, 'preprocess_data', 'train', 'train_Nan_date'), train_Nan_date)


def preprocess_test():
    # preprocess test data
    for i in range(5):
        print('---------------------------------')
        print(f'检验时段{i}')
        print('---------------------------------')
        print("验证数据-降水的缺失值\n", names['test_pre' + str(i)].isnull().sum())
        print('---------------------------------')
        print("验证数据-径流的缺失值\n", names['test_runoff' + str(i)].isnull().sum())
        print('---------------------------------')
        print("验证数据-预报降雨的缺失值\n", names['test_pre_predict' + str(i)].isnull().sum())
        print('---------------------------------')
        print("result: Nan is only found in train_runoff")
        print("preprocess method: average of train_runoff in a day")

        # search Nan index/date in test_runoff
        names['test_Nan_index' + str(i)] = names['test_runoff' + str(i)][names['test_runoff' + str(i)].isnull().values == True].index
        names['test_Nan_date' + str(i)] = pd.to_datetime(list(set(names['test_Nan_index' + str(i)].date)))

        # preprocess train data
        names['test_pre_D' + str(i)] = names['test_pre' + str(i)].resample("D").sum()  # sum-pre
        names['test_runoff_D' + str(i)] = names['test_runoff' + str(i)].resample("D").mean()  # mean-runoff

        # save
        names['test_pre_D' + str(i)].to_excel(os.path.join(home, 'preprocess_data', 'test', f'test_pre_D{i}.xlsx'))
        names['test_runoff_D' + str(i)].to_excel(os.path.join(home, 'preprocess_data', 'test', f'test_runoff_D{i}.xlsx'))
        names['test_pre_predict' + str(i)].to_excel(os.path.join(home, 'preprocess_data', 'test', f'test_pre_predict{i}.xlsx'))

        np.save(os.path.join(home, 'preprocess_data', 'test', f'test_Nan_index{i}'), names['test_Nan_index' + str(i)])
        np.save(os.path.join(home, 'preprocess_data', 'test', f'test_Nan_date{i}'), names['test_Nan_date' + str(i)])


if __name__ == '__main__':
    preprocess_train()
    preprocess_test()
