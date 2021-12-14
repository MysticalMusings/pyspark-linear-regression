from utils import *
import numpy as np
import time
import sys
from non_spark import linear_regression

N = 10000
D = 10
b = 100
# w = np.array([[5, 0.05], [1, 0.01], [20, 0.2], [1000, 1], [10000, 10]])
if D < 5:
    w = np.array([[5], [10], [20], [30], [40]])
    rng = np.array([[10, 3], [10, 3], [11, 2], [12, 2], [13, 2]])
else:
    w = np.random.normal(10, 5, (D, 1))
    rng = np.random.normal(5, 0.6, (D, 2))

generate_points(N, D, w, b, rng, delta=0.05)
