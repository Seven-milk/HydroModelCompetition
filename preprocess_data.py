# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# process_data(train and val data)
# hourly -> 3hourly: sum-pre/ mean-runoff
# out:
# train_pre/runoff_3H.xlsx
# val_pre_3H0/1/2/3/4.xlsx(5 val time) val_pre_predict3H0/1/2/3/4.xlsx val_runoff_D0/1/2/3/4.xlsx
import pandas as pd
import numpy as np
import os

# general set
home = 'H:/文件/水科学数值模拟/复赛/原始数据集'
obj_path = 'H:/文件/水科学数值模拟/复赛/数据预处理/preprocess_data'

# read train data, use xlsx index, follow section set DateTimeIndex
train_pre = pd.read_excel(os.path.join(home, 'train', '训练数据集-降雨.xlsx'), index_col=0)
train_runoff = pd.read_excel(os.path.join(home, 'train', '训练数据集-径流.xlsx'), index_col=0)
names = locals()

# set index for val data, 20 days for before pre and runoff, 2 days for predict-pre
'''假定都是从2017.01.01开始的检验数据
检验数据：给出数据20天，2017.01.01 - 2017.01.20
预报时间：2天预报降雨，2017.01.21-2017.01.22
'''
date_val = pd.date_range(start='2017-01-01 00:00:00', end='2017-01-20 23:00:00', freq='H')  # before pre and runoff
date_predict = pd.date_range(start='2017-01-21 00:00:00', end='2017-01-22 23:00:00', freq='H')  # predict-pre

# read val data
for i in range(5):
    names['val_pre' + str(i)] = pd.read_excel(os.path.join(home, 'val', '检验数据集-降雨.xlsx'), sheet_name=i,
                                               index_col=0, usecols=list(range(1, 22)))
    names['val_runoff' + str(i)] = pd.read_excel(os.path.join(home, 'val', '检验数据集-径流.xlsx'), sheet_name=i,
                                                  index_col=0, usecols=list(range(1, 6)))
    names['val_pre_predict' + str(i)] = pd.read_excel(os.path.join(home, 'val', '检验数据集-预报降雨.xlsx'),
                                                       sheet_name=i, index_col=0, usecols=list(range(21)))
    # set index
    names['val_pre' + str(i)].index = date_val
    names['val_runoff' + str(i)].index = date_val
    names['val_pre_predict' + str(i)].index = date_predict


def preprocess_train():
    # preprocess train data
    print('---------------------------------')
    print("训练数据-降水的缺失值\n", train_pre.isnull().sum())
    print('---------------------------------')
    print("训练数据-径流的缺失值\n", train_runoff.isnull().sum())
    print('---------------------------------')
    print("result: Nan is only found in train_runoff")
    print("preprocess method: average of train_runoff in 3H, sum pre in 3H")

    # search Nan index/date in train_runoff
    train_Nan_index = train_runoff[train_runoff.isnull().values == True].index
    train_Nan_date = pd.to_datetime(list(set(train_Nan_index.date)))

    # preprocess train data
    train_pre_3H = train_pre.resample("3H").sum()  # sum-pre
    train_runoff_3H = train_runoff.resample("3H").mean()  # mean-runoff

    # save
    train_pre_3H.to_excel(os.path.join(obj_path, 'train', 'train_pre_3H.xlsx'))
    train_runoff_3H.to_excel(os.path.join(obj_path, 'train', 'train_runoff_3H.xlsx'))

    np.save(os.path.join(obj_path, 'train', 'train_Nan_index'), train_Nan_index)
    np.save(os.path.join(obj_path, 'train', 'train_Nan_date'), train_Nan_date)


def preprocess_val():
    # preprocess val data
    for i in range(5):
        print('---------------------------------')
        print(f'检验时段{i}')
        print('---------------------------------')
        print("验证数据-降水的缺失值\n", names['val_pre' + str(i)].isnull().sum())
        print('---------------------------------')
        print("验证数据-径流的缺失值\n", names['val_runoff' + str(i)].isnull().sum())
        print('---------------------------------')
        print("验证数据-预报降雨的缺失值\n", names['val_pre_predict' + str(i)].isnull().sum())
        print('---------------------------------')
        print("result: Nan is only found in train_runoff")
        print("preprocess method: average of train_runoff in 3H, sum pre in 3H")

        # search Nan index/date in val_runoff
        names['val_Nan_index' + str(i)] = names['val_runoff' + str(i)][names['val_runoff' + str(i)].isnull().values == True].index
        names['val_Nan_date' + str(i)] = pd.to_datetime(list(set(names['val_Nan_index' + str(i)].date)))

        # preprocess val data
        names['val_pre_3H' + str(i)] = names['val_pre' + str(i)].resample("3H").sum()  # sum-pre
        names['val_runoff_3H' + str(i)] = names['val_runoff' + str(i)].resample("3H").mean()  # mean-runoff
        names['val_pre_predict_3H' + str(i)] = names['val_pre_predict' + str(i)].resample("3H").sum()  # sum-pre-predict

        # save
        names['val_pre_3H' + str(i)].to_excel(os.path.join(obj_path, 'val', f'val_pre_3H{i}.xlsx'))
        names['val_runoff_3H' + str(i)].to_excel(os.path.join(obj_path, 'val', f'val_runoff_3H{i}.xlsx'))
        names['val_pre_predict_3H' + str(i)].to_excel(os.path.join(obj_path, 'val', f'val_pre_predict_3H{i}.xlsx'))

        np.save(os.path.join(obj_path, 'val', f'val_Nan_index{i}'), names['val_Nan_index' + str(i)])
        np.save(os.path.join(obj_path, 'val', f'val_Nan_date{i}'), names['val_Nan_date' + str(i)])


def Nan_val():
    val_data = [path for path in os.listdir(os.path.join(obj_path, 'val')) if path.endswith('xlsx')]
    train_data = [path for path in os.listdir(os.path.join(obj_path, 'train')) if path.endswith('xlsx')]

    # val检验
    print('---------------------------------')
    print("val预处理后数据检验")
    for val_data_ in val_data:
        df = pd.read_excel(os.path.join(obj_path, 'val', val_data_), index_col=0)
        print('---------------------------------')
        print(f"{val_data_}-缺失值\n", df.isnull().sum())
        if df.isnull().sum().sum() == 0:
            print("result: 没有缺失值")
        else:
            print(f"result: 缺失{df.isnull().sum().sum()}个数据")
        print('---------------------------------')

    # train检验
    print('---------------------------------')
    print("train预处理后数据检验")
    for train_data_ in train_data:
        df = pd.read_excel(os.path.join(obj_path, 'train', train_data_), index_col=0)
        print('---------------------------------')
        print(f"{train_data_}-缺失值\n", df.isnull().sum())
        if df.isnull().sum().sum() == 0:
            print("result: 没有缺失值")
        else:
            print(f"result: 缺失{df.isnull().sum().sum()}个数据")
        print('---------------------------------')


if __name__ == '__main__':
    # preprocess_train()
    # preprocess_val()
    Nan_val()