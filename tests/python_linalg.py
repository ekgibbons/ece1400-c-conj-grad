import numpy as np
from scipy import io as spio

size_0 = np.random.randint(500,1000,size=1)[0]
size_1 = np.random.randint(250,500,size=1)[0]
A = np.random.uniform(-50, 50, size=(size_0, size_1))
b = np.random.uniform(-50, 50, size=(size_0, 1))
x = np.linalg.solve(A.T @ A ,A.T @ b).reshape(size_1,1)

spio.mmwrite("A_test",A)
spio.mmwrite("b_test",b)
spio.mmwrite("x_test",x)

