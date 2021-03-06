# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 使用方法：将实测结果放入*real_test/train.xlsx，将预测结果放入*predict_test/train.xlsx，运行即可

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import Nonparamfit
import re

target_real_train_path = [dir for dir in os.listdir() if dir.endswith('real_train.csv')][0]
target_real_test_path = [dir for dir in os.listdir() if dir.endswith('real_test.csv')][0]
target_predict_train_path = [dir for dir in os.listdir() if dir.endswith('predict_train.csv')][0]
target_predict_test_path = [dir for dir in os.listdir() if dir.endswith('predict_test.csv')][0]

target_real_train = pd.read_csv(target_real_train_path, index_col=False, header=None).values.T
target_real_test = pd.read_csv(target_real_test_path, index_col=False, header=None).values.T
target_predict_train = pd.read_csv(target_predict_train_path, index_col=False, header=None).values.T
target_predict_test = pd.read_csv(target_predict_test_path, index_col=False, header=None).values.T


# --------------------------- general evaluation ---------------------------
def general_evaluation(target_real_test, target_real_train, target_predict_test, target_predict_train, save_on='False',
                       batch=False, plot_on=True):
    # model evaluation
    def evaluation(target_test, target_predict_test, save_path="evaluation.xlsx", save_on=False, batch=False):
        NSE = 1 - sum((target_test - target_predict_test) ** 2) / sum((target_test - target_test.mean()) ** 2)
        PBIAS = sum(target_test - target_predict_test) * 100 / sum(target_test)
        RMSE = sum((target_test - target_predict_test) ** 2) ** 0.5
        RSR = RMSE / ((sum((target_test - target_test.mean()) ** 2)) ** 0.5)
        evaluation = np.vstack((NSE, PBIAS, RMSE, RSR))
        evaluation = pd.DataFrame(evaluation, columns=['Predict Hour' + str(i + 1) for i in range(NSE.shape[0])],
                                  index=['NSE', 'PBIAS', 'RMSE', 'RSR'])
        print(evaluation)

        # save evaluation_test
        if batch == False:
            save_on = input(f"是否保存 {save_path}? True or other:")

        if save_on == 'True':
            evaluation.to_excel(save_path)

        return evaluation

    # model evaluation_train
    evaluation_train = evaluation(target_real_train, target_predict_train, save_path="evaluation_train.xlsx",
                                  save_on=save_on, batch=batch)

    # evaluation test
    evaluation_test = evaluation(target_real_test, target_predict_test, save_path="evaluation_test.xlsx",
                                 save_on=save_on, batch=batch)

    # plot
    def plot_evaluation():
        # plot bar
        ax = plt.subplot(5, 1, 1)
        ticks_ = [f'第{i + 1}时段' for i in range(evaluation_train.shape[1])]
        labels_ = ['率定期', '验证期']
        bar_width = 0.2
        ticks_position = [i + (bar_width * 3) / 2 for i in range(evaluation_train.shape[1])]
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

        for i in range(evaluation_train.shape[1]):
            evaluation_test_text = "\n".join(
                [evaluation_test.index[j] + ': ' + '%.2f' % evaluation_test.iloc[j, i] for j in range(4)])
            x = len(target_predict_test[:, i]) * 0.84
            y = max(max(target_predict_test[:, i]), max(target_real_test[:, 0])) * 0.5

            ax = plt.subplot(5, 4, i + 5)
            ax.plot(target_predict_test[:, i], 'r', label="预测值")
            ax.plot(target_real_test[:, i], 'b', label="实测值")
            ax.legend(loc="upper left", prop={'family': 'SimHei'})
            ax.set_ylabel("runoff", fontdict={'family': 'Arial'})
            ax.set_xlabel("samples", loc='right', fontdict={'family': 'Arial'})
            ax.set_title(f'时段{i + 1}', fontdict={'family': 'SimHei'})
            ax.text(x, y, evaluation_test_text,
                    fontdict={'family': 'Arial', 'size': 8})  # , bbox=dict(facecolor='w', alpha=0.5)
            # ax.set_xlim(0, len(target_predict_test[:, i + 1]))

        # plot set
        plt.subplots_adjust(wspace=0.2, hspace=0.8)
        # plt.rcParams['font.sans-serif']=['SimHei']
        plt.show()

    # plot_evaluation
    if batch == True:
        plot_on = False
    if plot_on == True:
        plot_evaluation()


general_evaluation_on = input("是否进行general evaluation? True or Other:")
if general_evaluation_on == "True":
    general_evaluation(target_real_test, target_real_train, target_predict_test, target_predict_train, save_on=False,
                       batch=False, plot_on=True)


# --------------------------- extrame value evaluation ---------------------------
def extreme_evaluation(target_real_test, target_real_train, target_predict_test, target_predict_train, save_on=False,
                       save_path="evaluation_extreme.xlsx", plot_on='False', batch=False):
    # threshold select
    def threshold_select():
        target_real_all = np.vstack((target_real_test, target_real_train))
        threshold_high = []
        threshold_low = []

        for i in range(16):
            ed_distribution = Nonparamfit.EmpiricalDistribution()
            ed_distribution.fit(target_real_all[:, i], cdf_on=True)
            threshold_high.append(ed_distribution.ppf(0.8))
            threshold_low.append(ed_distribution.ppf(0.2))
            print(f"Hour {i + 1} threshold_high=", threshold_high[i], " ", "threshold_low=", threshold_low[i])

    # threshold_select()
    # result:
    # Hour 1 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 2 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 3 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 4 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 5 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 6 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 7 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 8 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 9 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 10 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 11 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 12 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 13 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 14 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 15 threshold_high= [0.25332798]   threshold_low= [0.04499276]
    # Hour 16 threshold_high= [0.25332798]   threshold_low= [0.04499276]

    threshold_high = 0.25332798
    threshold_low = 0.04499276

    def search_low_high(data, threshold_high, threshold_low):
        index_low = [i for i in range(len(data)) if data[i] < threshold_low]
        index_high = [i for i in range(len(data)) if data[i] > threshold_high]
        return index_low, index_high

    def plot_evaluation_low_high(target_real, target_predict, index_low, index_high):
        plt.figure()
        title = ["Train", "Test", "All"]
        for i in range(3):
            ax1 = plt.subplot(3, 3, 1 + i * 3)
            ax1.plot(target_real[i], "b", label="实测值")
            ax1.plot(target_predict[i], "r", label="预测值")
            ax1.legend(loc="upper left", prop={'family': 'SimHei'})
            ax1.set_title("全序列", fontdict={'family': 'SimHei'})
            ax1.set_ylabel(title[i])

            ax2 = plt.subplot(3, 3, 2 + i * 3)
            ax2.plot(target_real[i][index_high[i]], "b", label="实测值")
            ax2.plot(target_predict[i][index_high[i]], "r", label="预测值")
            ax2.legend(loc="upper left", prop={'family': 'SimHei'})
            ax2.set_title("高值序列", fontdict={'family': 'SimHei'})

            ax3 = plt.subplot(3, 3, 3 + i * 3)
            ax3.plot(target_real[i][index_low[i]], "b", label="实测值")
            ax3.plot(target_predict[i][index_low[i]], "r", label="预测值")
            ax3.legend(loc="upper left", prop={'family': 'SimHei'})
            ax3.set_title("低值序列", fontdict={'family': 'SimHei'})

        plt.subplots_adjust(wspace=0.2, hspace=0.5)

    # model evaluation_train
    def evaluation_low_high(target_real, target_predict, index_low, index_high, columns):
        target_real_low = target_real[index_low]
        target_real_high = target_real[index_high]
        target_predict_low = target_predict[index_low]
        target_predict_high = target_predict[index_high]

        NSE_all = 1 - sum((target_real - target_predict) ** 2) / sum((target_real - target_real.mean()) ** 2)
        NSE_high = 1 - sum((target_real_high - target_predict_high) ** 2) / sum(
            (target_real_high - target_real_high.mean()) ** 2)
        NSE_low = 1 - sum((target_real_low - target_predict_low) ** 2) / sum(
            (target_real_low - target_real_low.mean()) ** 2)

        evaluation = np.vstack((NSE_all, NSE_high, NSE_low))
        evaluation = pd.DataFrame(evaluation, index=['NSE_all', 'NSE_high', 'NSE_low'], columns=[columns])

        return evaluation

    # data
    target_real_all = np.vstack((target_real_test, target_real_train))
    target_predict_all = np.vstack((target_predict_test, target_predict_train))

    # evaluation
    evaluation_extreme_train = pd.DataFrame(
        columns=['Predict Hour' + str(i + 1) for i in range(target_real_all.shape[1])],
        index=['NSE_all', 'NSE_high', 'NSE_low'])
    evaluation_extreme_test = pd.DataFrame(
        columns=['Predict Hour' + str(i + 1) for i in range(target_real_all.shape[1])],
        index=['NSE_all', 'NSE_high', 'NSE_low'])
    evaluation_extreme_all = pd.DataFrame(
        columns=['Predict Hour' + str(i + 1) for i in range(target_real_all.shape[1])],
        index=['NSE_all', 'NSE_high', 'NSE_low'])

    for i in range(16):
        target_real_train_ = target_real_train[:, i]
        target_predict_train_ = target_predict_train[:, i]
        target_real_test_ = target_real_test[:, i]
        target_predict_test_ = target_predict_test[:, i]
        target_real_all_ = target_real_all[:, i]
        target_predict_all_ = target_predict_all[:, i]
        columns = 'Predict Hour' + str(i + 1)

        # train
        train_index_low, train_index_high = search_low_high(target_real_train_, threshold_high, threshold_low)
        evaluation_train = evaluation_low_high(target_real_train_, target_predict_train_, train_index_low,
                                               train_index_high, columns)
        evaluation_extreme_train.loc[:, 'Predict Hour' + str(i + 1)] = evaluation_train

        # test
        test_index_low, test_index_high = search_low_high(target_real_test_, threshold_high, threshold_low)
        evaluation_test = evaluation_low_high(target_real_test_, target_predict_test_, test_index_low, test_index_high,
                                              columns)
        evaluation_extreme_test.loc[:, 'Predict Hour' + str(i + 1)] = evaluation_test

        # all
        all_index_low, all_index_high = search_low_high(target_real_all_, threshold_high, threshold_low)
        evaluation_all = evaluation_low_high(target_real_all_, target_predict_all_, all_index_low, all_index_high,
                                             columns)
        evaluation_extreme_all.loc[:, 'Predict Hour' + str(i + 1)] = evaluation_all

        # plot
        if batch == False:
            plot_on = input("是否进行plot? True or other")

        if plot_on == 'True':
            target_real = [target_real_train_, target_real_test_, target_real_all_]
            target_predict = [target_predict_train_, target_predict_test_, target_predict_all_]
            index_low = [train_index_low, test_index_low, all_index_low]
            index_high = [train_index_high, test_index_high, all_index_high]
            plot_evaluation_low_high(target_real, target_predict, index_low, index_high)
            plt.show()

    print("evaluation_extreme_train\n", evaluation_extreme_train)
    print("evaluation_extreme_test\n", evaluation_extreme_test)
    print("evaluation_extreme_all\n", evaluation_extreme_all)

    # combine: 0-Train, 1-Test, 2-All
    evaluation_extreme_train.loc[:, "Name"] = pd.Series(np.full((3,), fill_value=0), index=evaluation_extreme_train.index)
    evaluation_extreme_test.loc[:, "Name"] = pd.Series(np.full((3,), fill_value=1), index=evaluation_extreme_test.index)
    evaluation_extreme_all.loc[:, "Name"] = pd.Series(np.full((3,), fill_value=2), index=evaluation_extreme_all.index)
    evaluation = pd.concat([evaluation_extreme_train, evaluation_extreme_test, evaluation_extreme_all])

    # save evaluation_test
    if batch == False:
        save_on = input("是否保存 evaluation_extreme.xlsx? True or other:")
    if save_on == True or 'True':
        evaluation.to_excel(save_path)


extreme_evaluation_on = input("是否进行extreme evaluation? True or Other:")
if extreme_evaluation_on == "True":
    extreme_evaluation(target_real_test, target_real_train, target_predict_test, target_predict_train)


# --------------------------- extrame value evaluation: BP ---------------------------
def extreme_evaluation_BP():
    home = 'H:\文件\水科学数值模拟\复赛\数据后处理\后处理数据\BP'
    target_predict_train_path = os.listdir(os.path.join(home, 'train'))
    target_predict_test_path = os.listdir(os.path.join(home, 'test'))
    for i in range(len(target_predict_train_path)):
        print(target_predict_train_path[i])
        target_predict_train_path_ = os.path.join(home, 'train', target_predict_train_path[i])
        target_predict_test_path_ = os.path.join(home, 'test', target_predict_test_path[i])
        target_predict_train = pd.read_csv(target_predict_train_path_, index_col=False, header=None).values.T
        target_predict_test = pd.read_csv(target_predict_test_path_, index_col=False, header=None).values.T
        save_path = os.path.join(home, 'BP' + re.search(r'\d+', target_predict_train_path[i])[0] + '_evaluation_extreme.xlsx')
        extreme_evaluation(target_real_test, target_real_train, target_predict_test, target_predict_train, save_on=True,
                           save_path=save_path, batch=True)


extreme_evaluation_BP_on = input("是否进行BP批量extreme evaluation? True or Other:")
if extreme_evaluation_BP_on == "True":
    extreme_evaluation_BP()


def plot_extreme_evaluation():
    home = 'H:\文件\水科学数值模拟\复赛\数据后处理\评估结果\BP_evaluation_extreme'
    evaluation_extreme_path = [path for path in os.listdir(home) if path.endswith('.xlsx')]

    NSE_all = np.zeros((len(evaluation_extreme_path), 17))
    NSE_high = np.zeros((len(evaluation_extreme_path), 17))
    NSE_low = np.zeros((len(evaluation_extreme_path), 17))
    BP_number = np.zeros((len(evaluation_extreme_path), ))
    for i in range(len(evaluation_extreme_path)):
        BP_number[i] = re.search(r'\d+', evaluation_extreme_path[i])[0]
        df = pd.read_excel(os.path.join(home, evaluation_extreme_path[i]), header=0, index_col=0)
        NSE_all[i, :] = df.values[3, :17]
        NSE_high[i, :] = df.values[4, :17]
        NSE_low[i, :] = df.values[5, :17]

    NSE_high = np.hstack((NSE_high, BP_number.reshape(-1, 1)))
    NSE_all = np.hstack((NSE_all, BP_number.reshape(-1, 1)))
    NSE_low = np.hstack((NSE_low, BP_number.reshape(-1, 1)))

    NSE_high = NSE_high[np.argsort(NSE_high[:, -1]), :]
    NSE_all = NSE_all[np.argsort(NSE_all[:, -1]), :]
    NSE_low = NSE_low[np.argsort(NSE_low[:, -1]), :]
    BP_number = sorted(BP_number)

    plt.figure()
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)

    ax1.set_xlabel("BP_number", loc='right')
    ax1.set_title("All")
    ax1.set_ylabel("NSE")
    ax2.set_xlabel("BP_number", loc='right')
    ax2.set_title("High")
    ax2.set_ylabel("NSE")
    ax3.set_xlabel("BP_number", loc='right')
    ax3.set_title("Low")
    ax3.set_ylabel("NSE")
    ax1.set_xlim(min(BP_number), max(BP_number))
    ax2.set_xlim(min(BP_number), max(BP_number))
    ax3.set_xlim(min(BP_number), max(BP_number))
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    for i in range(16):
        ax1.plot(BP_number, NSE_all[:, i], "r-", alpha=1 - i / 16, label='alpha: 16 Hour')
        ax1.plot(BP_number, NSE_all[:, i], "r.", alpha=1 - i / 16)
        ax2.plot(BP_number, NSE_high[:, i], "b-", alpha=1 - i / 16, label='alpha: 16 Hour')
        ax2.plot(BP_number, NSE_high[:, i], "b.", alpha=1 - i / 16)
        ax3.plot(BP_number, NSE_low[:, i], "g-", alpha=1 - i / 16, label='alpha: 16 Hour')
        ax3.plot(BP_number, NSE_low[:, i], "g.", alpha=1 - i / 16)
        if i == 0:
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper left")

    plt.subplots_adjust(hspace=0.5)
    plt.show()


extreme_evaluation_BP_plot_on = input("是否进行BP批量extreme evaluation Plot? True or Other:")
if extreme_evaluation_BP_plot_on == "True":
    plot_extreme_evaluation()









