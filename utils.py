import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os


def generate_points(N, D, w, b, rng, delta=0.03):
    normal = True

    def generator(normal=True):
        i = 0
        steps = (rng[:, 0] - rng[:, 1])/N
        x = np.zeros_like(rng[:, 0]).astype('float')
        # w_h = w[:, :1].copy()
        while i < N:
            if normal:
                for j in range(D):
                    x[j] = np.random.normal(rng[j, 0], rng[j, 1])
                    # w_h[j] = np.random.normal(w[j, 0], w[j, 1])
                y = w.T@x + b
            else:
                x = steps*i + rng[:, 0]
                y = w.T@x + b
            # noise = np.sqrt(w[:, 0].T@rng[:, 0]+b)
            y += np.random.normal(0, delta) * y
            yield y, x
            i += 1

    with open(f'points/{N}_{D}.txt', 'w') as f:
        for y, x in generator(normal):
            f.write(str(y[0])+' '+' '.join(x[:D].astype('str'))+'\n')


def plot_gradient(points, wb):
    def calc_loss(_w, _b):
        y_h = _w*x + _b
        return np.sum((y_h-y)**2)/y.shape[0]

    fig = plt.figure()  # 定义新的三维坐标轴
    ax = plt.axes(projection='3d')

    w = wb[:, 0]
    b = wb[:, 1]
    y = points[:, 0]
    x = points[:, 1]

    # 定义三维数据
    ww = np.linspace(3*w.min()-w.max(), 3*w.max()-w.min(), 100)
    bb = np.linspace(3*b.min()-b.max(), 3*b.max()-b.min(), 100)
    # ww = np.linspace(w.min(), w.max(), 100)
    # bb = np.linspace(b.min(), b.max(), 100)
    W, B = np.meshgrid(ww, bb)
    L = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            L[i, j] = calc_loss(W[i, j], B[i, j])

    # 作图
    ax.set_xlabel('W')
    ax.set_ylabel('b')
    ax.set_zlabel('loss')
    ax.plot_surface(W, B, L, cmap='rainbow')
    # 等高线图，要设置offset，为Z的最小值
    # ax.contour(W, B, L, zdim='z', cmap='rainbow')

    # zd = 13*np.random.random(100)
    # xd = 5*np.sin(zd)
    # yd = 5*np.cos(zd)
    # ax.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图

    l = np.zeros_like(w)
    for i in range(l.size):
        l[i] = calc_loss(w[i], b[i])

    # 防止与曲面重合（试试效果
    offset = (l.max()-l.min())*0.02
    l += offset
    ax.plot3D(w, b, l, color='r')  # 绘制空间曲线
    plt.show()


def plot_loss(L):
    def check_monotone(loss):
        stop_time = None
        is_monotone = True
        for i in range(loss.size-1):
            if loss[i] < loss[i+1]:
                is_monotone = False
            if is_monotone and int(loss[i]) == int(loss[i+1]):
                stop_time = i
        return is_monotone, stop_time
    max = -1
    plt.figure()
    max_stop_time = -1
    max_size = -1
    for label in L.keys():
        loss = L[label][0]
        # 限制xy轴坐标范围，避免梯度爆炸的情况
        is_monotone, stop_time = check_monotone(loss)
        if loss.size > max_size:
            max_size = loss.size
        if is_monotone:
            if loss.max() > max:
                max = loss.max()
            if stop_time:
                if stop_time > max_stop_time:
                    max_stop_time = stop_time
            else:
                stop_time = loss.size + 1
        plt.plot([i for i in range(1, loss.size+1)],
                 loss, L[label][1], label=label)
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ylim((0, max+0.1*max))
    plt.xlim((0, max_stop_time + int((max_size-max_stop_time)*0.3)))
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_line(points, wb):
    yd = points[:, 0].T
    xd = points[:, 1].T
    w = wb[-1, :-1]
    b = wb[-1, -1]
    x = np.linspace(xd.min(), xd.max(), 3)
    y = w*x + b
    plt.scatter(xd, yd, cmap='Blues')  # 绘制散点图
    plt.plot(x, y, 'k-')  # 绘制空间曲线
    plt.show()


def data_fromtxt(file):
    strs = list(open(file, 'r').read().split('\n'))
    if strs[-1] == '':
        strs = strs[:-1]
    D = len(strs[1].split(' '))
    matrix = np.zeros((len(strs), D))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(
            s.replace(',', ' '), dtype=np.float32, sep=' ')
    return np.array(matrix)


if __name__ == '__main__':
    # points = data_fromtxt('D:\\download\\10000_1.txt')
    # wb = data_fromtxt('D:\\download\\yarn_100_weights.txt')
    # points = data_fromtxt(sys.argv[1])
    # wb = data_fromtxt('results/weights.txt')
    # plot_line(points, wb)
    # plot_gradient(points, wb)
    plot_loss({"标准化": [data_fromtxt('results/loss_with_stdize.txt'), 'b-'],
              "非标准化": [data_fromtxt('results/loss_without_stdize.txt'), 'r-']})
