import numpy as np
import matplotlib.pyplot as plt
import time
from utils import points_fromtxt
import sys


def mathod_result(d):
    x = d[:, 1:]
    b = d[:, :1]
    A = np.hstack((x, np.ones_like(b)))
    return np.linalg.inv(A.T@A)@A.T@b


def standardize(x, mean, std):
    return (x-mean)/std


def recover(x, w, b, mean, std):
    return x*std+mean, w/std, b - np.sum(w*mean/std)


def linear_regression(data, iterations=100, lr=0.004, stdize=False, mathod=False):
    x = data[:, 1:].T
    y = data[:, :1]
    n = data.shape[0]

    mean = None
    std = None
    if stdize:
        mean = np.mean(x, 1).reshape(x.shape[0], 1)
        std = np.std(x, 1).reshape(x.shape[0], 1)
        x = standardize(x, mean, std)

    D = x.shape[0]
    w_h = np.array([[0.5]]*D)
    b_h = 0
    y_h = np.zeros_like(y)

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        y_h = x.T@w_h + b_h
        dw = x@(y_h-y)
        db = np.sum(y_h-y)
        w_h -= dw*lr/n
        b_h -= db*lr/n
        print("w:\n", w_h)
        print('b:\n', b_h)
        # print('w is', w_h, 'b is', b_h, 'loss:', np.sum((y_h-y)**2)/n)

    if stdize:
        x, w_h, b_h = recover(x, w_h, b_h, mean, std)

    print('final w is\n', w_h)
    print('final b is\n', b_h)
    if mathod:
        r = mathod_result(data)
        print('\nmathematic:\nw is', r[:-1], '\nb is', r[-1])
    # plot(data, np.array([x.T, x.T@w_h + b_h]))


# x = np.random.normal(0, 1, [2, n])
# Weights = np.array([[3, 10]]).reshape(2, 1)
# y = x.T@Weights+5
# data = np.hstack([y, x.T])

start = time.time()

data = points_fromtxt(sys.argv[1])
linear_regression(data, int(sys.argv[2]), lr=0.02, stdize=1, mathod=1)
end = time.time()
print("time: " + str(end - start)+' s')
