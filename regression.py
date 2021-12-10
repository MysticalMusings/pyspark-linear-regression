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


def gradient_w(matrix, w, b):
    Y = matrix[:, :1]
    X = matrix[:, 1:].T
    Y_h = X.T@w + b
    return (X@(Y_h - Y))


def gradient_b(matrix, w, b):
    Y = matrix[:, :1]
    X = matrix[:, 1:].T
    Y_h = X.T@w + b
    return np.sum(Y_h - Y)


def add(x, y):
    x += y
    return x


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
    w = 2 * np.random.ranf(size=(D, 1)) - 1
    b = 2 * np.random.ranf() - 1

    print("Initial w: " + str(w))
    print("Initial b: " + str(b))

    start = time.time()

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        dw = points.map(lambda m: gradient_w(m, w, b)).reduce(add).reshape(D, 1)
        db = points.map(lambda m: gradient_b(m, w, b)).reduce(add)
        w -= dw*lr/N
        b -= db*lr/N
        print("w:\n", w)
        print('b:\n', b)

    end = time.time()

    print("Final w:\n" + str(w))
    print("Final b:\n" + str(b))
    print("time: " + str(end - start)+' s')

    spark.stop()
