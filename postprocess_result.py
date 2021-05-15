# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

home = 'F:/文件/水科学数值模拟大赛/prelim/attribute_target'
# attribute_test = np.load(os.path.join(home, 'attribute_test.npy'))
# target_test = np.load(os.path.join(home, 'target_test.npy'))
# target_predict = np.load('F:/文件/水科学数值模拟大赛/prelim/model1/target_predict.npy')
target_test = np.load(os.path.join(home, 'target_test.npy'))
target_predict = np.load('F:/文件/水科学数值模拟大赛/prelim/model2/target_predict.npy')

# model evaluation
NSE = 1 - sum((target_test - target_predict)**2) / sum((target_test - target_test.mean())**2)
PBIAS = sum(target_test - target_predict) * 100 / sum(target_test)
RMSE = sum((target_test - target_predict)**2) ** 0.5
RSR = RMSE / ((sum((target_test - target_test.mean())**2)) ** 0.5)

evaluation = np.vstack((NSE, PBIAS, RMSE, RSR))

evaluation = pd.DataFrame(evaluation, columns=['Predict Day' + str(i+1) for i in range(7)], index=['NSE', 'PBIAS', 'RMSE', 'RSR'])
evaluation.to_excel('evaluation.xlsx')

# plot
ax = plt.subplot(4, 1, 1)
ax.plot(target_predict[:, 0], 'r', label="predict")
ax.plot(target_test[:, 0], 'b', label="real data")
ax.legend(loc="upper left")
# ax.set_xlabel("days")
ax.set_ylabel("runoff")
ax.set_title(f'Predict Day{1}')

for i in range(6):
    ax = plt.subplot(4, 2, i + 3)
    ax.plot(target_predict[:, i + 1], 'r', label="predict")
    ax.plot(target_test[:, i + 1], 'b', label="real data")
    ax.legend(loc="upper left")
    # ax.set_xlabel("days")
    ax.set_ylabel("runoff")
    ax.set_title(f'Predict Day{i + 2}')

plt.subplots_adjust(wspace=0.2, hspace=0.8)
plt.show()