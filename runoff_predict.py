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
home = 'F:/文件/水科学数值模拟大赛/prelim/attribute_target'
attribute_train = np.load(os.path.join(home, 'attribute_train.npy'))
target_train = np.load(os.path.join(home, 'target_train.npy'))

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


runoffmodel = RunoffFCNN()
runoffmodel.compile(optimizer='adam', loss="mse", metrics=['accuracy'])
history = runoffmodel.fit(attribute_train, target_train, batch_size=32, epochs=1000, validation_split=0.2,
                          validation_freq=1)
runoffmodel.summary()

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