import numpy as np
import matplotlib.pyplot as plt
import time
from utils import data_fromtxt
import sys
from scipy import stats


def mathod_result(d):
    x = d[:, 1:]
    b = d[:, :1]
    A = np.hstack((x, np.ones_like(b)))
    return np.linalg.inv(A.T@A)@A.T@b


def loss(y_h, y):
    return np.sum((y_h-y)**2)/y.shape[0]


def standardize(x, mean, std):
    return (x-mean)/std


def recover(w, b, mean, std):
    return w/std, b - np.sum(w*mean/std)


def linear_regression(data, iterations=100, lr=0.004, stdize=False, mathod=False):
    x = data[:, 1:].T
    y = data[:, :1]
    n = data.shape[0]
    D = x.shape[0]

    mean = None
    std = None
    if stdize:
        mean = np.mean(x, 1).reshape(D, 1)
        std = np.std(x, 1).reshape(D, 1)
        x = standardize(x, mean, std)

    w = np.array([[0.5]]*D)
    b = 0
    y_h = np.zeros_like(y)

    f = open('results/weights.txt', 'w')
    f2 = open('results/loss.txt', 'w')
    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        y_h = x.T@w + b
        dw = x@(y_h-y)
        db = np.sum(y_h-y)
        w -= dw*lr/n
        b -= db*lr/n
        if stdize:
            tmp_w, tmp_b = recover(w, b, mean, std)
            L = loss(data[:, 1:]@tmp_w + tmp_b, y)
        else:
            tmp_w = w.reshape(1, D)
            tmp_b = b
            L = loss(y_h, y)
        f.write(' '.join(str(x)
                for x in tmp_w.reshape(tmp_w.size)) + f' {tmp_b}\n')
        f2.write(str(L)+'\n')
        print("w:\n", w)
        print('b:\n', b)
        print('loss: ', L)

    f.close()
    if stdize:
        w, b = recover(w, b, mean, std)
        y_h = data[:, 1:]@w + b

    print('final w is\n', w)
    print('final b is\n', b)
    print('loss:', loss(y_h, y))
    if mathod:
        r = mathod_result(data)
        print('least square:\nw is', r[:-1], '\nb is', r[-1])
        w = r[:-1]
        b = r[-1]
        y_h = data[:, 1:]@w + b
        print('loss: ', loss(y_h, y))
        # 用scipy验证结果
        print(stats.linregress(data[:, 1:].reshape(
            data[:, 1:].size), data[:, :1].reshape(data[:, :1].size)))
    return w, b


if __name__ == '__main__':
    start = time.time()
    data = data_fromtxt(sys.argv[1])
    w, b = linear_regression(
        data, int(sys.argv[2]), lr=0.02, stdize=1, mathod=1)
    end = time.time()
    print("time: " + str(end - start)+' s')
