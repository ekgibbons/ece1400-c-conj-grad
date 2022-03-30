import numpy as np
from scipy import io as spio

size_mat_0 = 2048
size_mat_1 = 1024
A = np.random.uniform(-10, 10, size=(size_mat_0, size_mat_1))
b = np.random.uniform(-10, 10, size=(size_mat_0,1))

# A = np.random.randint(0, 20, size=(5,3))
# b = np.random.randint(0, 20, size=(5,1))

x = np.linalg.solve(A.T@A,A.T@b).reshape(-1,1)

spio.mmwrite("A_test",A)
spio.mmwrite("b_test",b)
spio.mmwrite("x_test",x)
 
