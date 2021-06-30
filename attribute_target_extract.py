# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# extract attribute and target datasets for model

import numpy as np
import pandas as pd
import os

# general set
home = 'H:/文件/水科学数值模拟/复赛/数据预处理/preprocess_data'
obj_path = 'H:/文件/水科学数值模拟/复赛/数据预处理/attribute_target'
name = locals()


# ---------------------------- train data ----------------------------
# read train data
train_pre = pd.read_excel(os.path.join(home, 'train', 'train_pre_3H.xlsx'), index_col=0)
train_runoff = pd.read_excel(os.path.join(home, 'train', 'train_runoff_3H.xlsx'), index_col=0)

# train data sample: 17361 ~ (8:2) = 13888 : 3473
'''0-17535：共17536——>有效样本：17361
target: train_runoff：【160(160 before): 17520(15 predict, 16-1)】=17361 损失信息：0~159+17521~17535=20*8=160+15=175
    每个样本16 (2D * 8 = 16H) * 1 section
attribute3: train_pre：同train_runoff：【160(160 before): 17520(15 predict, 16-1)】=17361  损失信息：175
    每个样本16 (2D * 8 = 16H) * 20 station = 320
attribute1: train_pre: 【0: 17520-160=17360】= 17360 损失信息 17520~17535=16
    每个样本160 (20D * 8 = 160) * 20 station = 3200
attribute2: train_runoff:【0: 17520-160=17360】= 17360 损失信息  17520~17535=16
    每个样本160 (20D * 8 = 160) * 4 station = 640
'''
# n(len(train_runoff) - 160(before) - 16(predict) + 1(first) == 17361, samples) * m(16, targets),
# target
target = np.zeros((len(train_runoff) - 160 - 15, 16))
# n(len(train_runoff) - 160(before) - 16(predict) + 1(first) == 17361, samples) * m(16, predict-days) * k(20, stations),
# precipitation-predict
attribute3 = np.zeros((len(train_runoff) - 160 - 15, 16, 20))
# n(len(train_runoff) - 160(len) - 16(predict) + 1(first) == 17361, samples) * m(160, before-days) * k(20, stations),
# precipitation-before
attribute1 = np.zeros((len(train_runoff) - 160 - 15, 160, 20))
# n(len(train_runoff) - 160(len) - 16(predict) + 1(first) == 17361, samples) * m(160, before-days) * k(4, stations),
# runoff-before
attribute2 = np.zeros((len(train_runoff) - 160 - 15, 160, 4))

# sample = 17361

# extract and put into attribute and target
for i in range(160, len(train_runoff) - 16 + 1):
    target[i - 160, :] = train_runoff.iloc[i:i + 16, 3].values
    attribute3[i - 160, :, :] = train_pre.iloc[i:i + 16, :].values
    attribute1[i - 160, :, :] = train_pre.iloc[i - 160: i, :].values
    attribute2[i - 160, :, :] = train_runoff.iloc[i - 160: i, :].values


# save_before_reshape: all npy
def save_before_reshape():
    np.save(os.path.join(obj_path, 'train', 'attribute1_before_reshape_train'), attribute1[:13889, :, :])
    np.save(os.path.join(obj_path, 'train', 'attribute2_before_reshape_train'), attribute2[:13889, :, :])
    np.save(os.path.join(obj_path, 'train', 'attribute3_before_reshape_train'), attribute3[:13889, :, :])
    np.save(os.path.join(obj_path, 'train', 'attribute1_before_reshape_test'), attribute1[13889:, :, :])
    np.save(os.path.join(obj_path, 'train', 'attribute2_before_reshape_test'), attribute2[13889:, :, :])
    np.save(os.path.join(obj_path, 'train', 'attribute3_before_reshape_test'), attribute3[13889:, :, :])

    np.save(os.path.join(obj_path, 'train', 'target_train'), target[:13889, :])
    np.save(os.path.join(obj_path, 'train', 'target_test'), target[13889:, :])


save_before_reshape()

# reshape and save as csv: all csv
attribute1 = attribute1.reshape((17361, 3200))
attribute2 = attribute2.reshape((17361, 640))
attribute3 = attribute3.reshape((17361, 320))
attribute1 = pd.DataFrame(attribute1,
                          columns=[f"pre_beforeHour{160 - i}_station{j + 1}" for i in range(160) for j in range(20)])
attribute2 = pd.DataFrame(attribute2,
                          columns=[f"runoff_beforeHour{160 - i}_station{j + 1}" for i in range(160) for j in range(4)])
attribute3 = pd.DataFrame(attribute3,
                          columns=[f"pre_predictHour{i + 1}_station{j + 1}" for i in range(16) for j in range(20)])
target = pd.DataFrame(target, columns=[f"runoff_predictHour{i + 1}" for i in range(16)])

# combine all attribute
attribute = attribute1.join(attribute2)
attribute = attribute.join(attribute3)


# save after reshape: all csv
def save_after_reshape():
    # train:[:13889]
    attribute1.iloc[:13889, :].to_csv(os.path.join(obj_path, 'train', 'attribute1_after_reshape_train.csv'))
    attribute2.iloc[:13889, :].to_csv(os.path.join(obj_path, 'train', 'attribute2_after_reshape_train.csv'))
    attribute3.iloc[:13889, :].to_csv(os.path.join(obj_path, 'train', 'attribute3_after_reshape_train.csv'))
    target.iloc[:13889, :].to_csv(os.path.join(obj_path, 'train', 'target_train.csv'))
    attribute.iloc[:13889, :].to_csv(os.path.join(obj_path, 'train', 'attribute_combine_after_reshape_train.csv'))

    # test:[13889:]
    attribute1.iloc[13889:, :].to_csv(os.path.join(obj_path, 'train', 'attribute1_after_reshape_test.csv'))
    attribute2.iloc[13889:, :].to_csv(os.path.join(obj_path, 'train', 'attribute2_after_reshape_test.csv'))
    attribute3.iloc[13889:, :].to_csv(os.path.join(obj_path, 'train', 'attribute3_after_reshape_test.csv'))
    target.iloc[13889:, :].to_csv(os.path.join(obj_path, 'train', 'target_test.csv'))
    attribute.iloc[13889:, :].to_csv(os.path.join(obj_path, 'train', 'attribute_combine_after_reshape_test.csv'))


save_after_reshape()

# attribute combine save as npy
attribute_ = attribute.values


def save_attribute_npy():
    np.save(os.path.join(obj_path, 'train', 'attribute_combine_after_reshape_train'), attribute_[:13889, :])
    np.save(os.path.join(obj_path, 'train', 'attribute_combine_after_reshape_test'), attribute_[13889:, :])


save_attribute_npy()
# ---------------------------- val data ----------------------------
# 5 val dataset
for i in range(5):
    val_pre_3H = pd.read_excel(os.path.join(home, 'val', f'val_pre_3H{i}.xlsx'), index_col=0)
    val_pre_predict_3H = pd.read_excel(os.path.join(home, 'val', f'val_pre_predict_3H{i}.xlsx'), index_col=0)
    val_runoff_3H = pd.read_excel(os.path.join(home, 'val', f'val_runoff_3H{i}.xlsx'), index_col=0)
    attribute1 = val_pre_3H.values
    attribute2 = val_runoff_3H.values
    attribute3 = val_pre_predict_3H.values

    np.save(os.path.join(obj_path, 'val', f'attribute1_val{i}'), attribute1)
    np.save(os.path.join(obj_path, 'val', f'attribute2_val{i}'), attribute2)
    np.save(os.path.join(obj_path, 'val', f'attribute3_val{i}'), attribute3)