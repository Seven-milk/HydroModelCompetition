# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import re
import pandas as pd
import os
import matplotlib.pyplot as plt

# general set
home = 'F:/文件/水科学数值模拟大赛/prelim/数据预处理/attribute_target'
attribute_train = np.load(os.path.join(home, 'attribute_train.npy'))[:1732]
target_train = np.load(os.path.join(home, 'target_train.npy'))[:1732]
attribute_test = np.load(os.path.join(home, 'attribute_test.npy'))
target_test = np.load(os.path.join(home, 'target_test.npy'))
model_path = 'F:/文件/水科学数值模拟大赛/prelim/model1'
checkpoint_save_path = os.path.join(model_path, 'checkpoint/mnist.ckpt')

# shuffle train data
np.random.seed(10)
np.random.shuffle(attribute_train)
np.random.seed(10)
np.random.shuffle(target_train)
tf.random.set_seed(10)


class RunoffFCNN(Model):
    ''' Fully Connected Neural Network '''

    def __init__(self):
        super(RunoffFCNN, self).__init__()
        self.Dense1 = tf.keras.layers.Dense(66, activation="sigmoid")
        self.Drop1 = tf.keras.layers.Dropout(0.2)
        self.Dense2 = tf.keras.layers.Dense(7, activation="relu")

    def call(self, x):
        x = self.Dense1(x)
        x = self.Drop1(x)
        y = self.Dense2(x)
        return y


# model set
runoffmodel = RunoffFCNN()
runoffmodel.compile(optimizer='adam', loss="mse", metrics=['accuracy'])


# load model
if os.path.exists(checkpoint_save_path + '.index'):
    print('------------------- load model -------------------')
    runoffmodel.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)


# model fit
history = runoffmodel.fit(attribute_train, target_train, batch_size=32, epochs=1000, validation_split=0.2,
                      validation_freq=1, callbacks=[cp_callback])

# plot
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1, 2, 1)
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Training and Validation Loss")

plt.legend()
plt.show()

runoffmodel.summary()
print(runoffmodel.trainable_variables)
with open(os.path.join(model_path, 'weights.txt'), 'w') as file:
    for v in runoffmodel.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')


# predict
target_predict_test = runoffmodel.predict(attribute_test)
# np.save("target_predict_test", target_predict_test)
df = pd.DataFrame(target_predict_test)
df.to_excel("target_predict_test_FCNN.xlsx")

attribute_train = np.load(os.path.join(home, 'attribute_train.npy'))[:1732]
target_predict_train = runoffmodel.predict(attribute_train)
# np.save("target_predict_train", target_predict_train)
df = pd.DataFrame(target_predict_train)
df.to_excel("target_predict_train_FCNN.xlsx")


# predict plot
def predict_plot():
    predict_num = 16
    fig, ax = plt.subplots(4, 4, sharex=True, sharey=False)
    ax = ax.flatten()
    for i in range(predict_num):
        rand_ = np.random.randint(0, len(target_train))
        predict_ = runoffmodel.predict(attribute_train[rand_, :].reshape(1, 620))
        ax[i].plot(predict_.reshape(7, ), 'r', label="predict")
        ax[i].plot(target_train[rand_, :], 'b', label="real data")
        print("mse: ", sum((predict_ - target_train[rand_, :]).flatten()**2))
        ax[i].legend(loc="upper right")
        ax[i].set_xlabel("days")
        ax[i].set_ylim(0, 1)

predict_plot()