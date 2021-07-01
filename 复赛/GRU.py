# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
import math
import matplotlib.pyplot as plt
import pandas as pd


# pretreatment dataset
class TrainDataset(Dataset):
    ''' TrainDataset '''
    def __init__(self):
        # general set
        home = 'H:/文件/水科学数值模拟/复赛/数据预处理/attribute_target/train'

        # read dataset
        self.attribute1 = np.load(os.path.join(home, 'attribute1_before_reshape_train.npy')).astype(np.float32)
        self.attribute2 = np.load(os.path.join(home, 'attribute2_before_reshape_train.npy')).astype(np.float32)
        self.attribute3 = np.load(os.path.join(home, 'attribute3_before_reshape_train.npy')).astype(np.float32)
        self.target = np.load(os.path.join(home, 'target_train.npy')).astype(np.float32)

        self.runoff = self.attribute2
        self.pre = np.concatenate((self.attribute1, self.attribute3), axis=1)

        # len
        self.len = len(self.target)

    def __getitem__(self, item):
        return self.runoff[item, :, :], self.pre[item, :, :], self.target[item, :]

    def __len__(self):
        return self.len


class TestDataset(Dataset):
    ''' TestDataset '''
    def __init__(self):
        # general set
        home = 'H:/文件/水科学数值模拟/复赛/数据预处理/attribute_target/train'

        # read dataset
        self.attribute1 = np.load(os.path.join(home, 'attribute1_before_reshape_test.npy')).astype(np.float32)
        self.attribute2 = np.load(os.path.join(home, 'attribute2_before_reshape_test.npy')).astype(np.float32)
        self.attribute3 = np.load(os.path.join(home, 'attribute3_before_reshape_test.npy')).astype(np.float32)
        self.target = np.load(os.path.join(home, 'target_test.npy')).astype(np.float32)

        self.runoff = self.attribute2
        self.pre = np.concatenate((self.attribute1, self.attribute3), axis=1)

        # len
        self.len = len(self.target)

    def __getitem__(self, item):
        return self.runoff[item, :, :], self.pre[item, :, :], self.target[item, :]

    def __len__(self):
        return self.len


class ValDataset(Dataset):
    ''' TestDataset '''
    def __init__(self, i):
        ''' i: val dataset number: 0, 1, 2, 3, 4 '''
        # general set
        home = 'H:/文件/水科学数值模拟/复赛/数据预处理/attribute_target/val'

        # read dataset
        self.attribute1 = np.load(os.path.join(home, f'attribute1_val{i}.npy')).astype(np.float32)
        self.attribute2 = np.load(os.path.join(home, f'attribute2_val{i}.npy')).astype(np.float32)
        self.attribute3 = np.load(os.path.join(home, f'attribute3_val{i}.npy')).astype(np.float32)

        self.attribute1 = self.attribute1.reshape((1, *self.attribute1.shape))
        self.attribute2 = self.attribute2.reshape((1, *self.attribute2.shape))
        self.attribute3 = self.attribute3.reshape((1, *self.attribute3.shape))

        self.runoff = self.attribute2
        self.pre = np.concatenate((self.attribute1, self.attribute3), axis=1)

        # len
        self.len = 1

    def __getitem__(self, item):
        return self.runoff[item, :, :], self.pre[item, :, :]

    def __len__(self):
        return self.len


# build model
class Model03(torch.nn.Module):

    def __init__(self):
        super(Model03, self).__init__()
        self.gru_pre = torch.nn.GRU(input_size=20, hidden_size=1, batch_first=True)
        self.gru_runoff = torch.nn.GRU(input_size=4, hidden_size=16, batch_first=True)
        self.linear = torch.nn.Linear(in_features=32, out_features=16, bias=True)
        self.activate = torch.nn.Sigmoid()

    def forward(self, runoff, pre):
        out_pre, _ = self.gru_pre(pre)
        out_pre = out_pre[:, -16:, :]
        _, hidden_runoff = self.gru_runoff(runoff)

        hidden_runoff = hidden_runoff.view(-1, 16)
        out_pre = out_pre.view(-1, 16)
        x = torch.cat([hidden_runoff, out_pre], dim=1)
        x = self.activate(self.linear(x))
        return x


# train
def train(epoch, train_loader, optimizer, model, criterion, print_batch_num, since):
    epoch_loss = 0.0
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        runoff, pre, target = data
        optimizer.zero_grad()

        # forward
        output = model(runoff, pre)
        loss = criterion(output, target)

        # backward
        loss.backward()
        optimizer.step()

        # cal running_loss & epoch_loss
        epoch_loss += loss.item()
        running_loss += loss.item()

        # print loss for each print_batch_num
        if batch_idx % print_batch_num == 0:
            # time print
            time_str = time_since(since)
            print("[epoch%d, batch%5d] loss: %.8f training_time: " % (epoch + 1, batch_idx + 1, running_loss /
                                                                      print_batch_num) + time_str)
            running_loss = 0.0

    return epoch_loss


# nse
def NSE(predict, real):
    nse = 1 - sum((real - predict)**2) / sum((real - predict.mean())**2)
    return nse


# test
def test(epoch, test_loader, model):
    predict = np.zeros((1, 16))
    real = np.zeros((1, 16))
    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            runoff, pre, target = data
            predict_ = model(runoff, pre)
            real = np.vstack((real, target.numpy()))
            predict = np.vstack((predict, predict_.numpy()))

    real = real[1:]
    predict = predict[1:]
    nse = NSE(predict, real)
    print(f"epoch{epoch} NSE= {nse}\n")
    return nse


# plot nse
def plot_train_test(epoch_list, loss_list, nse_list):
    # plot set
    fig, ax = plt.subplots(1, 2)
    ax_ = ax[1]
    ax = ax[0]
    ax.grid(True)
    ax_.grid(True)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax_.set_xlabel("Epoch")
    ax_.set_ylabel("NSE")

    nse_list = [sum(nse_)/len(nse_) for nse_ in nse_list]
    ax.plot(epoch_list, loss_list, "b", label="Loss")
    ax_.plot(epoch_list, nse_list, "r", label="NSE")
    plt.show()


# time
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


# train model
def train_cycle_model(epochs=100):
    # this running set
    batch_size = 32
    print_batch_num = 5

    # read data
    train_dataset = TrainDataset()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # model
    model = Model03()

    # optimizer and loss function
    criterion = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # checkpoint
    path_checkpoint_read = "H:/文件/水科学数值模拟/复赛/GRU/checkpoint/ckpt_best_model01.pkl"
    if os.path.isfile(path_checkpoint_read):
        # model read
        checkpoint = torch.load(path_checkpoint_read)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        # loss list(train), nse list(test), epoch list
        loss_list = checkpoint['loss_list']
        nse_list = checkpoint['nse_list']
        epoch_list = checkpoint['epoch_list']

    else:
        # init
        start_epoch = 0

        # loss list(train), nse list(test), epoch list
        loss_list = []
        nse_list = []
        epoch_list = []

    # time
    since = time.time()

    # train and test for each epoch
    for epoch in list(range(start_epoch + 1, start_epoch + 2 + epochs)):
        epoch_list.append(epoch)
        epoch_loss = train(epoch, train_loader, optimizer, model, criterion, print_batch_num, since)  # train
        loss_list.append(epoch_loss)
        nse = test(epoch, test_loader, model)  # test
        nse_list.append(nse)

        # check point: each 50 epoches or loss decrease > 5%
        if (epoch % 50 == 0) or (len(loss_list) > 2 and loss_list[-2] - loss_list[-1] / loss_list[-2] > 5):
            checkpoint = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_list": epoch_list,
                "loss_list": loss_list,
                "nse_list": nse_list,
                "epoch": epoch_list[-1],
            }
            torch.save(checkpoint, f"H:/文件/水科学数值模拟/复赛/GRU/checkpoint/ckpt_best_model01_{epoch}.pkl")

    # plot
    plot_train_test(epoch_list, loss_list, nse_list)

    # save model
    checkpoint = {
        "net": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch_list": epoch_list,
        "loss_list": loss_list,
        "nse_list": nse_list,
        "epoch": epoch_list[-1],
    }
    torch.save(checkpoint, "H:/文件/水科学数值模拟/复赛/GRU/checkpoint/ckpt_best_model01.pkl")
    torch.save(model, 'H:/文件/水科学数值模拟/复赛/GRU/model01.pth')


# model overview
def overview_model():
    checkpoint = torch.load('H:/文件/水科学数值模拟/复赛/GRU/checkpoint/ckpt_best_model01.pkl')
    epoch_list = checkpoint["epoch_list"]
    loss_list = checkpoint["loss_list"]
    nse_list = checkpoint["nse_list"]
    epoch = checkpoint["epoch"]
    net = checkpoint["net"]

    # print
    print(f"parameters are {net}")
    print(f"Now the train epoch is {epoch}")

    # plot
    plot_train_test(epoch_list, loss_list, nse_list)


# predict
def predict(runoff, pre):
    '''
    input:
        runoff: samples * 160 times * 4 stations
        pre: samples * 176 times * 20 stations
    '''
    model = torch.load('H:/文件/水科学数值模拟/复赛/GRU/model01.pth')
    runoff = torch.from_numpy(runoff)
    pre = torch.from_numpy(pre)
    predict_out = model(runoff, pre)
    return predict_out


# predict_test
def predict_test(save_on=True):
    model = torch.load('H:/文件/水科学数值模拟/复赛/GRU/model01.pth')
    test_dataset = TestDataset()
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, num_workers=1)
    predict = np.zeros((1, 16))
    with torch.no_grad():
        for data in test_loader:
            runoff, pre, target_ = data
            predict_ = model(runoff, pre)
            predict = np.vstack((predict, predict_.numpy()))

    predict = predict[1:]
    predict = predict.T

    if save_on == True:
        df = pd.DataFrame(predict)
        df.to_csv("predict_test.csv", index=False, header=False)

    return predict


# predict_train
def predict_train(save_on=True):
    model = torch.load('H:/文件/水科学数值模拟/复赛/GRU/model01.pth')
    train_dataset = TrainDataset()
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, num_workers=1)
    predict = np.zeros((1, 16))
    with torch.no_grad():
        for data in train_loader:
            runoff, pre, target_ = data
            predict_ = model(runoff, pre)
            predict = np.vstack((predict, predict_.numpy()))

    predict = predict[1:]
    predict = predict.T

    if save_on == True:
        df = pd.DataFrame(predict)
        df.to_csv("predict_train.csv", index=False, header=False)

    return predict


# predict_val
def predict_val(save_on=True):
    model = torch.load('H:/文件/水科学数值模拟/复赛/GRU/model01.pth')
    predict_all = np.zeros((16, 5))
    for i in range(5):
        val_dataset = ValDataset(i)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False, num_workers=1)
        predict = np.zeros((1, 16))
        with torch.no_grad():
            for data in val_loader:
                runoff, pre = data
                predict_ = model(runoff, pre)
                predict = np.vstack((predict, predict_.numpy()))

        predict = predict[1:]
        predict = predict.T
        predict = predict.reshape((-1, ))
        predict_all[:, i] = predict

    if save_on == True:
        df = pd.DataFrame(predict_all)
        df.to_csv(f"predict_val.csv", index=False, header=False)


if __name__ == '__main__':
    # train_cycle_model(epochs=5)
    overview_model()
    predict_test()
    predict_train()
    predict_val()