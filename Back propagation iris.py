# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:12:52 2019

@author: lizhenping
"""

import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from sklearn.preprocessing import MinMaxScaler
import random
def load_dataset(dataset_path, n_train_data):
    """加载数据集，对数据进行预处理，并划分训练集和验证集
    :param dataset_path: 数据集文件路径
    :param n_train_data: 训练集的数据量
    :return: 划分好的训练集和验证集
    """
    dataset = []
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    with open(dataset_path, 'r') as file:
        # 读取CSV文件，以逗号为分隔符
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            # 将字符串类型的特征值转换为浮点型
            row[0:4] = list(map(float, row[0:4]))
            # 将标签替换为整型
            row[4] = label_dict[row[4]]
            # 将处理好的数据加入数据集中
            dataset.append(row)

    # 对数据进行归一化处理
    dataset = np.array(dataset)
    mms = MinMaxScaler()
    for i in range(dataset.shape[1] - 1):
        dataset[:, i] = mms.fit_transform(dataset[:, i].reshape(-1, 1)).flatten()

    # 将类标转为整型
    dataset = dataset.tolist()
    for row in dataset:
        row[4] = int(row[4])
    # 打乱数据集
    random.shuffle(dataset)

    # 划分训练集和验证集
    train_data = dataset[0:n_train_data]
    val_data = dataset[n_train_data:]

    return train_data, val_data




plt.style.use('classic')
np.set_printoptions(precision=3, suppress=True)

X=-1 + 2*np.random.random((20,4))



n_train_data = 130
file_path = './iris.csv'
train_data, val_data = load_dataset(file_path, n_train_data)
train_data = np.array(train_data)
# 训练模型
print("test")
#network = train(train_data, l_rate, epochs, n_hidden, val_data)
X=[]
X= train_data[:,0:4]
y = train_data[:,4:5]
print(y)
y = np.squeeze(y)
y = y.T
y=y.astype(int)
Y = np.eye(3)[y]



LEARNING_RATE = 2.0
NUM_EPOCHS = 40




def stable_softmax(z):
  # z is 3 x 1
  a = np.exp(z - max(z)) / np.sum(np.exp(z - max(z)))
  # a is 3 x 1
  return a


def forward_propagate(x, W, b):
  # W is 3 x 2
  # x is 2 x 1
  # b is 3 x 1
  z = np.matmul(W, x) + b
  a = stable_softmax(z)
  # z is 3 x 1
  # a is 3 x 1
  return z, a


def get_loss(y, a):
  return -1 * np.sum(y * np.log(a))

def get_loss_numerically_stable(y, z):
   return -1 * np.sum(y * (z + (-z.max() - np.log(np.sum(np.exp(z-z.max()))))))

def get_gradients(x, z, a, y):
  da = (-y / a)

  matrix = np.matmul(a, np.ones((1, 3))) * (np.identity(3) - np.matmul(np.ones((3, 1)), a.T))
  dz = np.matmul(matrix, da)

  dW = dz * x.T
  db = dz.copy()

  return dz, dW, db

def gradient_descent(W, b, dW, db, learning_rate):
  W = W - learning_rate * dW
  b = b - learning_rate * db
  return W, b

# random initialization
W_initial = np.random.rand(3, 4)
W = W_initial.copy()
b = np.zeros((3, 1))

W_cache = []
b_cache = []
L_cache = []

for i in range(NUM_EPOCHS):
  dW = np.zeros(W.shape)
  db = np.zeros(b.shape)
  L = 0
  for j in range(X.shape[0]):
    X.shape[0]
    x_j = X[j,:].reshape(4,1)
    y_j = Y[j,:].reshape(3,1)

    z_j, a_j = forward_propagate(x_j, W, b)
    loss_j = get_loss_numerically_stable(y_j, z_j)
    _, dW_j, db_j = get_gradients(x_j, z_j, a_j, y_j)

    dW += dW_j
    db += db_j
    L += loss_j

  dW *= (1.0/130)
  db *= (1.0/130)
  L *= (1.0/130)

  W, b = gradient_descent(W, b, dW, db, LEARNING_RATE)

  W_cache.append(W)
  b_cache.append(b)
  L_cache.append(L)

plt.grid()
plt.title('Loss', size=18)
plt.xlabel('Number of iterations', size=15)
plt.ylabel('Loss', size=15)
plt.ylim([0, max(L_cache) * 1.1])
plt.plot(L_cache)
plt.show()

plt.savefig('image.png')

plt.close()
plt.clf()
plt.cla()