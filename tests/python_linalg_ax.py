import numpy as np
from scipy import io as spio

size_rand = np.random.randint(7,11,size=(2))
A = np.random.normal(loc=10,scale=5,size=size_rand)
x = np.random.normal(loc=10,scale=5,size=size_rand[1])
sol = A @ x

spio.mmwrite("A_dax",A)
spio.mmwrite("x_dax",x[:,None])
spio.mmwrite("sol_dax",sol[:,None])

