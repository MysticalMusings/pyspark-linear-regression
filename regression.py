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
    param = 2 * np.random.ranf(size=(D+1, 1)) - 1

    print("Initial param:\n" + str(param))

    start = time.time()

    for i in range(iterations):
        print("On iteration %i" % (i + 1))
        grad = points.map(lambda m: gradient(m, param)
                          ).reduce(add).reshape(D+1, 1)
        param -= grad*lr/N
        print("param:\n", param)

    end = time.time()

    print("Final param:\n" + str(param))
    print("time: " + str(end - start)+' s')

    spark.stop()
