import numpy as np
import matplotlib.pyplot as plt
import time


D = 2


def init_data(w, b, n):
    xs = np.arange(-10, 10, 20/n)
    ys = np.zeros_like(xs)
    for i in range(n):
        ys[i] = w*xs[i] + b + (np.random.random()-0.35)/3*(w*xs[i] + b)
    return np.array([xs, ys])


def init_fromtxt(file):
    strs = list(open(file, 'r').read().split('\n'))
    matrix = np.zeros((len(strs), D+1))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(
            s.replace(',', ' '), dtype=np.float32, sep=' ')
    return np.array(matrix)


def plot(d, d_h):
    plt.plot(d[:, 1:], d[:, 0], 'ro')
    plt.plot(d_h[0], d_h[1], 'b-')
    plt.show()


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

    w_h = np.array([[0.5], [0.5]])
    b_h = 0
    y_h = np.zeros_like(y)

    start = time.time()
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
        print('mathematic: w is', r[:-1], 'b is', r[-1])
    # plot(data, np.array([x.T, x.T@w_h + b_h]))


# x = np.random.normal(0, 1, [2, n])
# Weights = np.array([[3, 10]]).reshape(2, 1)
# y = x.T@Weights+5
# data = np.hstack([y, x.T])
data = init_fromtxt('points.txt')

linear_regression(data, 10, lr=0.02, stdize=1, mathod=1)
