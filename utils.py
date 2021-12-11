import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os


def generate_points(N, D):
    w = np.array([[5], [1], [20], [1000], [10000]])
    b = 100
    rng = np.array([[200, 20], [100, 10], [1, 0.1], [200, 20], [1000, 60]])
    normal = True

    def generator(normal=True):
        i = 0
        steps = (rng[:, 0] - rng[:, 1])/N
        x = np.zeros_like(rng[:, 0]).astype('float')
        while i < N:
            if normal:
                for j in range(D):
                    x[j] = np.random.normal(rng[j, 0], rng[j, 1])
                y = w.T@x + b
            else:
                x = steps*i + rng[:, 0]
                y = w.T@x + b
            y += np.random.normal(0, 1)*sy
            yield y, x
            i += 1

    with open('points/{0}_{1}.txt'.format(N, D), 'w') as f:
        for y, x in generator(normal):
            # y += y * (random.random() * 2 - 1)/4
            f.write(str(y[0])+' '+' '.join(x[:D].astype('str'))+'\n')
        NEWLINE_SIZE_IN_BYTES = 2  # 2 on Windows?
        f.seek(0, os.SEEK_END)  # Go to the end of the file.
        # Go backwards one byte from the end of the file.
        f.seek(f.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
        f.truncate()  # Truncate the file to this point.


def plot_gradient(points, W, B):
    fig = plt.figure()  # 定义新的三维坐标轴
    ax = plt.axes(projection='3d')

    # 定义三维数据
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(X)+np.cos(Y)

    # 作图
    ax.plot_surface(X, Y, Z, cmap='rainbow')
    # 等高线图，要设置offset，为Z的最小值
    ax.contour(X, Y, Z, zdim='z', offset=-2, cmap='rainbow')

    z = np.linspace(0, 13, 1000)
    x = 5*np.sin(z)
    y = 5*np.cos(z)
    zd = 13*np.random.random(100)
    xd = 5*np.sin(zd)
    yd = 5*np.cos(zd)
    ax.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
    ax.plot3D(x, y, z, 'gray')  # 绘制空间曲线
    plt.show()


def plot_line(points_path, wb_path):
    points = points_fromtxt(points_path)
    w, b = wb_fromtxt(wb_path)
    yd = points[:, 0].T
    xd = points[:, 1].T
    x = np.linspace(xd.min(), xd.max(), 1000)
    y = w*x + b
    plt.scatter(xd, yd, cmap='Blues')  # 绘制散点图
    plt.plot(x, y, 'k-')  # 绘制空间曲线
    plt.show()


def points_fromtxt(file):
    strs = list(open(file, 'r').read().split('\n'))
    D = len(strs[1].split(' '))
    matrix = np.zeros((len(strs), D))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(
            s.replace(',', ' '), dtype=np.float32, sep=' ')
    return np.array(matrix)


def wb_fromtxt(file):
    pass


if __name__ == '__main__':
    # x = data[:, 1:].T
    # y = data[:, :1]
    # n = data.shape[0]
    # y_h = x.T@w_h + b_h
    # loss = np.sum((y_h-y)**2)/n
    plot_line(sys.argv[1], 9.72408131, 263.7276463358178)
