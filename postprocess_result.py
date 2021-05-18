# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 使用方法：将实测结果放入target_test.xlsx，将预测结果放入target_predict.xlsx，运行即可

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


target_test = pd.read_excel('target_test.xlsx', index_col=0).values
target_predict = pd.read_excel('target_predict.xlsx', index_col=0).values

# model evaluation
NSE = 1 - sum((target_test - target_predict)**2) / sum((target_test - target_test.mean())**2)
PBIAS = sum(target_test - target_predict) * 100 / sum(target_test)
RMSE = sum((target_test - target_predict)**2) ** 0.5
RSR = RMSE / ((sum((target_test - target_test.mean())**2)) ** 0.5)
evaluation = np.vstack((NSE, PBIAS, RMSE, RSR))
evaluation = pd.DataFrame(evaluation, columns=['Predict Day' + str(i+1) for i in range(7)], index=['NSE', 'PBIAS',
                                                                                                   'RMSE', 'RSR'])
print(evaluation)

# save evaluation
evaluation_save = input("是否保存evaluation.xlsx(True or False):")
if evaluation_save == True:
    evaluation.to_excel('evaluation.xlsx')


# plot
for i in range(7):
    evaluation_text = "\n".join([evaluation.index[j] + ': ' + '%.2f' % evaluation.iloc[j, i] for j in range(4)])
    x = len(target_predict[:, i]) * 0.9
    y = max(max(target_predict[:, i]), max(target_test[:, 0])) * 0.5

    ax = plt.subplot(4, 2, i + 1)
    ax.plot(target_predict[:, i], 'r', label="预测值")
    ax.plot(target_test[:, i], 'b', label="实测值")
    ax.legend(loc="upper left", prop={'family': 'SimHei'})
    ax.set_ylabel("runoff", fontdict={'family': 'Arial'})
    ax.set_xlabel("samples", loc='right', fontdict={'family': 'Arial'})
    ax.set_title(f'Day{i + 1}')
    ax.text(x, y, evaluation_text, fontdict={'family': 'Arial', 'size': 8})
    # ax.set_xlim(0, len(target_predict[:, i + 1]))

# plot hist
ax = plt.subplot(4, 2, 8)
ticks_ = [f'第{i+1}天' for i in range(7)]
labels_ = [evaluation.index[i] for i in range(4)]
bar_width = 0.2
ticks_position = [i + (bar_width * 3) / 2 for i in range(7)]
x = list(range(len(ticks_)))

for i in range(4):
    x_ = [xx + bar_width * i for xx in x]
    bar_value = [evaluation.iloc[i, j] for j in range(len(x))]
    ax.bar(x_, bar_value, width=bar_width, label=labels_[i])

ax.legend(loc="lower right", prop={'family': 'Arial', 'size': 8}, ncol=4)
ax.set_title("评价指标", fontdict={'family': 'SimHei'})
ax.set_xticks(ticks_position)
ax.set_xticklabels(ticks_, fontdict={'family': 'SimHei'})
ax.set_ylabel("评价指标值", fontdict={'family': 'SimHei'})

plt.subplots_adjust(wspace=0.2, hspace=0.8)
# plt.rcParams['font.sans-serif']=['SimHei']
plt.show()