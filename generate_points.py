import random
import os
from typing import final
import numpy as np


N = 10
w = np.array([[3], [5]])
b = 10
rng = np.array([[0, 3], [5, 3]])
normal = 1


def generator(N, w, b, rng, normal=True):
    i = 0
    x_1st = rng[:, 0]
    x_2nd = rng[:, 1]
    steps = (x_2nd - x_1st)/N
    x = np.zeros_like(x_1st).astype('float')
    while i < N:
        if normal:
            for j in range(x.shape[0]):
                x[j] = np.random.normal(x_1st[j], x_2nd[j])
            y = w.T@x + b
        else:
            x = steps*i + x_1st
            y = w.T@x + b
            y += (np.random.random()-0.35)/3*y
        yield y, x
        i += 1


with open('points.txt', 'w') as f:
    for y, x in generator(N, w, b, rng, normal):
        # y += y * (random.random() * 2 - 1)/4
        f.write(str(y[0])+' '+' '.join(x.astype('str'))+'\n')
    NEWLINE_SIZE_IN_BYTES = 2  # 2 on Windows?
    f.seek(0, os.SEEK_END)  # Go to the end of the file.
    # Go backwards one byte from the end of the file.
    f.seek(f.tell() - NEWLINE_SIZE_IN_BYTES, os.SEEK_SET)
    f.truncate()  # Truncate the file to this point.
