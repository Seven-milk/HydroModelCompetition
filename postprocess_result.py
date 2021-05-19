# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 使用方法：将实测结果放入target_test.xlsx，将预测结果放入target_predict_test.xlsx，运行即可

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


target_test = pd.read_excel('target_test.xlsx', index_col=0).values
target_predict_test = pd.read_excel('target_predict_test.xlsx', index_col=0).values
target_train = pd.read_excel('target_train.xlsx', index_col=0).values
target_predict_train = pd.read_excel('target_predict_train.xlsx', index_col=0).values

# model evaluation test
NSE = 1 - sum((target_test - target_predict_test)**2) / sum((target_test - target_test.mean())**2)
PBIAS = sum(target_test - target_predict_test) * 100 / sum(target_test)
RMSE = sum((target_test - target_predict_test)**2) ** 0.5
RSR = RMSE / ((sum((target_test - target_test.mean())**2)) ** 0.5)
evaluation_test = np.vstack((NSE, PBIAS, RMSE, RSR))
evaluation_test = pd.DataFrame(evaluation_test, columns=['Predict Day' + str(i+1) for i in range(7)],
                               index=['NSE', 'PBIAS', 'RMSE', 'RSR'])
print(evaluation_test)

# save evaluation_test
evaluation_test_save = input("是否保存evaluation_test.xlsx(True or False):")
if evaluation_test_save == True:
    evaluation_test.to_excel('evaluation_test.xlsx')

# model evaluation_train
NSE = 1 - sum((target_train - target_predict_train)**2) / sum((target_train - target_train.mean())**2)
PBIAS = sum(target_train - target_predict_train) * 100 / sum(target_train)
RMSE = sum((target_train - target_predict_train)**2) ** 0.5
RSR = RMSE / ((sum((target_train - target_train.mean())**2)) ** 0.5)
evaluation_train = np.vstack((NSE, PBIAS, RMSE, RSR))
evaluation_train = pd.DataFrame(evaluation_train, columns=['Predict Day' + str(i+1) for i in range(7)], index=['NSE', 'PBIAS',
                                                                                                   'RMSE', 'RSR'])
print(evaluation_train)

# save evaluation_test
evaluation_train_save = input("是否保存evaluation_train.xlsx(True or False):")
if evaluation_train_save == True:
    evaluation_train.to_excel('evaluation_train.xlsx')

# plot
for i in range(7):
    evaluation_test_text = "\n".join([evaluation_test.index[j] + ': ' + '%.2f' % evaluation_test.iloc[j, i] for j in range(4)])
    x = len(target_predict_test[:, i]) * 0.9
    y = max(max(target_predict_test[:, i]), max(target_test[:, 0])) * 0.5

    ax = plt.subplot(4, 2, i + 1)
    ax.plot(target_predict_test[:, i], 'r', label="预测值")
    ax.plot(target_test[:, i], 'b', label="实测值")
    ax.legend(loc="upper left", prop={'family': 'SimHei'})
    ax.set_ylabel("runoff", fontdict={'family': 'Arial'})
    ax.set_xlabel("samples", loc='right', fontdict={'family': 'Arial'})
    ax.set_title(f'Day{i + 1}')
    ax.text(x, y, evaluation_test_text, fontdict={'family': 'Arial', 'size': 8})
    # ax.set_xlim(0, len(target_predict_test[:, i + 1]))

# plot bar
ax = plt.subplot(4, 2, 8)
ticks_ = [f'第{i+1}天' for i in range(7)]
labels_ = ['率定期', '验证期']
bar_width = 0.2
ticks_position = [i + (bar_width * 3) / 2 for i in range(7)]
x = list(range(len(ticks_)))

x_ = x
bar_value = [evaluation_train.iloc[0, j] for j in range(len(x))]
ax.bar(x_, bar_value, width=bar_width, label=labels_[0])

x_ = [xx + bar_width for xx in x]
bar_value = [evaluation_test.iloc[0, j] for j in range(len(x))]
ax.bar(x_, bar_value, width=bar_width, label=labels_[1])

ax.legend(loc="upper right", prop={'family': 'SimHei', 'size': 8}, ncol=4)
ax.set_title("模型评估", fontdict={'family': 'SimHei'})
ax.set_xticks(ticks_position)
ax.set_xticklabels(ticks_, fontdict={'family': 'SimHei'})
ax.set_ylabel("NSE", fontdict={'family': 'Arial'})
ax.set_ylim(0.6, 1)

plt.subplots_adjust(wspace=0.2, hspace=0.8)
# plt.rcParams['font.sans-serif']=['SimHei']
plt.show()