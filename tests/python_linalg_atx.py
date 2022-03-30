import numpy as np
from scipy import io as spio

size_rand = np.random.randint(7,11,size=(2))
x = np.random.normal(loc=10,scale=5,size=size_rand[0])
A = np.random.normal(loc=10,scale=5,size=size_rand)

sol = A.T@x

spio.mmwrite("A_atx",A)
spio.mmwrite("x_atx",x[:,None])
spio.mmwrite("sol_atx",sol[:,None])
             

