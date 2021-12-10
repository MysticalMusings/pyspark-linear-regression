import sys
import numpy as np
from pyspark.sql import SparkSession
import time

# 维数
D = 2
# 学习率
lr = 0.02


def readPointBatch(iterator):
    strs = list(iterator)
    matrix = np.zeros((len(strs), D + 1))
    for i, s in enumerate(strs):
        matrix[i] = np.fromstring(
            s.replace(',', ' '), dtype=np.float32, sep=' ')
    return [matrix]


def gradient(matrix, param):
    Y = matrix[:, :1]
    X = matrix[:, 1:].T
    X = np.vstack([X, np.ones((1, X.shape[1]))])
    Y_h = X.T@param
    return (X@(Y_h - Y)).sum(1)


def add(x, y):
    x += y
    return x


def standardize(matrix):
    Y = matrix[:, :1]
    X = matrix[:, 1:].T
    X_std = (X-mean) / std
    return np.hstack([Y, X_std.T])


def recover(w, b):
    return w/std, b - np.sum(w*mean/std)


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: linear_regression <file> <Number of samples> <iterations>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("linear_regression")\
        .getOrCreate()

    points = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])\
        .mapPartitions(readPointBatch).cache()

    print(points.count())
    iterations = int(sys.argv[3])
    N = int(sys.argv[2])
    # param = 2 * np.random.ranf(size=(D+1, 1)) - 1
    # 测试使用参数
    param = np.array([[0.5], [0.5], [0]])

    # 标准化
    mean = points.map(lambda m: m[:, 1:].sum(0)).sum()/N
    std = np.sqrt(points.map(lambda m: ((m[:, 1:]-mean)**2).sum(0)).sum()/N)
    mean = mean.reshape(D, 1)
    std = std.reshape(D, 1)
    points = points.map(lambda m: standardize(m))

    print("Initial param:\n" + str(param))

    start = time.time()

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        grad = points.map(lambda m: gradient(m, param)
                          ).reduce(add).reshape(D+1, 1)
        param -= grad*lr/N
        print("param:\n", param, '\n')
    end = time.time()

    w, b = recover(param[:-1], param[-1])
    print("Final w:\n" + str(w))
    print("Final b:\n" + str(b))
    print("time: " + str(end - start)+' s')

    spark.stop()
