import sys
import numpy as np
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
import time


class PySparkLR:
    def __init__(self, input, mode, iterations=10):
        if mode == "":
            mode = 'default'
        self.mode = mode
        base = os.path.basename(input)
        filename = os.path.splitext(base)[0]
        self.input = input
        try:
            iterations = int(iterations)
        except:
            iterations = 10

        # 去除mode参数中的特殊字符
        modeInPath = mode.replace('[', '').replace(']', '').replace('*', 'A')
        self.outputTimePath = '/results/{}/{}_{}_time.txt'.format(
            filename, modeInPath, iterations)
        self.outputWeightsPath = '/results/{}/{}_{}_weights.txt'.format(
            filename, modeInPath, iterations)
        self.iterations = iterations
        self.N = int(filename.split('_')[0])
        self.D = int(filename.split('_')[1])

    def readPointBatch(self, iterator):
        strs = list(iterator)
        matrix = np.zeros((len(strs), self.D + 1))
        rowToDelete = []
        for i, s in enumerate(strs):
            try:
                matrix[i] = np.fromstring(
                    s.replace(',', ' '), dtype=np.float32, sep=' ')
            except:
                rowToDelete.append(i)
        matrix = np.delete(matrix, rowToDelete, axis=0)
        return [matrix]

    def gradient(self, matrix, param):
        Y = matrix[:, :1]
        X = matrix[:, 1:].T
        X = np.vstack([X, np.ones((1, X.shape[1]))])
        Y_h = X.T@param
        return (X@(Y_h - Y)).sum(1)

    def standardize(self, points):
        def process(matrix):
            Y = matrix[:, :1]
            X = matrix[:, 1:].T
            X_std = (X-self.mean) / self.std
            return np.hstack([Y, X_std.T])
        mean = points.map(lambda m: m[:, 1:].sum(0)).sum()/self.N
        std = np.sqrt(points.map(lambda m: (
            (m[:, 1:]-mean)**2).sum(0)).sum()/self.N)
        self.mean = mean.reshape(self.D, 1)
        self.std = std.reshape(self.D, 1)
        return points.map(lambda m: process(m)).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    def recover(self, param):
        param = param.copy()
        w = param[:-1].copy()
        b = param[-1].copy()
        param[:-1] = w/self.std
        param[-1] = b - np.sum(w*self.mean/self.std)
        return param

    def linearRegression(self, lr=0.02, standardization=True):
        start = time.time()

        conf = SparkConf().setAppName("linear_regression")
        # if self.mode != "default" and self.mode != "yarn":
        #     conf.setMaster(self.mode)
        sc = SparkContext(conf=conf).getOrCreate()

        Path = sc._jvm.org.apache.hadoop.fs.Path
        FileSystem = sc._jvm.org.apache.hadoop.fs.FileSystem

        # create FileSystem and Path objects
        hadoopConfiguration = sc._jsc.hadoopConfiguration()
        hadoopFs = FileSystem.get(hadoopConfiguration)

        # create datastream and write out file
        outputTimeStream = hadoopFs.create(Path(self.outputTimePath))
        outputWeightsStream = hadoopFs.create(Path(self.outputWeightsPath))

        points = sc.textFile(self.input).map(
            lambda r: r[0]).mapPartitions(self.readPointBatch).cache()

        # param = 2 * np.random.ranf(size=(D+1, 1)) - 1
        # 测试使用参数
        param = np.zeros((self.D+1, 1))
        param[:-1] = np.array([[0.5]]*self.D)

        # 标准化
        if standardization:
            points = self.standardize(points)

        timeStr = "{}:\t{} s\n"
        checkpoint = time.time()
        prepTime = timeStr.format("prepTime", str(checkpoint-start))
        print(prepTime)
        outputTimeStream.write(prepTime.encode('utf-8'))

        print("Initial param:\n" + str(param))

        # 迭代
        for i in range(self.iterations):
            print("On iteration %i" % (i + 1))
            grad = points.map(lambda m: self.gradient(m, param)
                              ).reduce(lambda x, y: x+y).reshape(self.D+1, 1)
            param -= grad*lr/self.N
            print("param:\n", param, '\n')
            if standardization:
                tmp = self.recover(param)
            else:
                tmp = param
            outputWeightsStream.write(' '.join(str(x)
                                      for x in tmp.reshape(tmp.size)).encode('utf-8'))
            outputWeightsStream.write('\n'.encode('utf-8'))

        end = time.time()
        iterateTime = timeStr.format("iterateTime", str(end-checkpoint))
        print(iterateTime)
        outputTimeStream.write(iterateTime.encode('utf-8'))

        if standardization:
            param = self.recover(param)
        print("Final w:\n" + str(param[:-1]))
        print("Final b:\n" + str(param[-1]))
        print()

        totalTime = timeStr.format("totalTime", str(end-start))
        print(totalTime)
        outputTimeStream.write(totalTime.encode('utf-8'))
        outputTimeStream.write(
            ("Final w:\n" + str(param[:-1])).encode('utf-8'))
        outputTimeStream.write(
            ("\nFinal b:\n" + str(param[-1])).encode('utf-8'))

        outputTimeStream.close()
        outputWeightsStream.close()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: linear_regression <input file> [<iterations> <mode>]", file=sys.stderr)
        sys.exit(-1)

    lr = 0.02

    try:
        mode = sys.argv[3]
    except:
        mode = ''

    PySparkLR(sys.argv[1], iterations=sys.argv[2],
              mode=mode).linearRegression(lr=lr, standardization=True)
