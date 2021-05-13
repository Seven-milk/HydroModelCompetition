# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# extract attribute and target datasets for model

import numpy as np
import pandas as pd
import os

# general set
home = "F:/文件/水科学数值模拟大赛/prelim/preprocess_data"
# name = locals()

# read data
train_pre = pd.read_excel(os.path.join(home, 'train', 'train_pre_D.xlsx'), index_col=0)
train_runoff = pd.read_excel(os.path.join(home, 'train', 'train_runoff_D.xlsx'), index_col=0)

# train data sample: 2166
'''0-2191：共2192——>有效样本：2166
target: train_runoff：【20~20+7: 2085~2085+7】=2166 损失信息：0~19=20
attribute3: train_pre：同train_runoff：【20~20+7: 2085~2085+7】=2166  损失信息：0~19=20

attribute1: train_pre: 【0: 0+20~2165+20】=2166 损失信息 2185~2191=7
attribute2: train_runoff:【0: 0+20~2165+20】=2166 损失信息  2185~2191=7
'''
# n(len(train_runoff) - 20(before) - 7(len(7)) == 2166, samples) * m(7, targets),
# target
target = np.zeros((len(train_runoff) - 20 - 6, 7))
# n(len(train_runoff) - 20(before) - 6(len(7) - 1) == 2166, samples) * m(20, before-days) * k(20, stations),
# precipitation-before
attribute1 = np.zeros((len(train_runoff) - 20 - 6, 20, 20))
# n(len(train_runoff) - 20(before) - 7(len(7) - 1) == 2166, samples) * m(20, before-days) * k(4, stations),
# runoff-before
attribute2 = np.zeros((len(train_runoff) - 20 - 6, 20, 4))
# n(len(train_runoff) - 20(before) - 7(len(7) - 1) == 2166, samples) * m(7, predict-days) * k(20, stations),
# precipitation-predict
attribute3 = np.zeros((len(train_runoff) - 20 - 6, 7, 20))

# extract and reshape
for i in range(20, len(train_runoff) - 7 + 1):
    target[i - 20, :] = train_runoff.iloc[i:i + 7, 3].values
    attribute3[i - 20, :, :] = train_pre.iloc[i:i + 7, :].values
    attribute1[i - 20, :, :] = train_pre.iloc[i - 20: i, :].values
    attribute2[i - 20, :, :] = train_runoff.iloc[i - 20: i, :].values

# post process
attribute1 = attribute1.reshape((2166, 400))
attribute2 = attribute2.reshape((2166, 80))
attribute3 = attribute3.reshape((2166, 140))
attribute1 = pd.DataFrame(attribute1,
                          columns=[f"pre_beforeday{20 - i}_station{j + 1}" for i in range(20) for j in range(20)])
attribute2 = pd.DataFrame(attribute2,
                          columns=[f"runoff_beforeday{20 - i}_station{j + 1}" for i in range(20) for j in range(4)])
attribute3 = pd.DataFrame(attribute3,
                          columns=[f"pre_predictday{i + 1}_station{j + 1}" for i in range(7) for j in range(20)])
target = pd.DataFrame(target, columns=[f"runoff_predictday{i + 1}" for i in range(7)])

attribute = attribute1.join(attribute2)
attribute = attribute.join(attribute3)


# save
def save_attribute():
    attribute1.to_excel('attribute1.xlsx')
    attribute2.to_excel('attribute2.xlsx')
    attribute3.to_excel('attribute3.xlsx')
    target.to_excel('target_train.xlsx')
    attribute.to_excel('attribute_train.xlsx')
    target.iloc[1732:, :].to_excel('target_test.xlsx')
    attribute.iloc[1732:, :].to_excel('attribute_test.xlsx')


# attribute/target_train
attribute_train = attribute.values
target_train = target.values


def save_attribute_target_tarin():
    np.save('attribute_train', attribute_train)
    np.save('target_train', target_train)


# attribute/target_test
attribute_test = attribute_train[1732:, :]
target_test = target_train[1732:, :]


def save_attribute_target_test():
    np.save('attribute_test', attribute_test)
    np.save('target_test', target_test)

