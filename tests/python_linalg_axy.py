import numpy as np
from scipy import io as spio

size_rand = np.random.randint(7,11,size=(1))
x = np.random.normal(loc=10,scale=5,size=size_rand[0])
y = np.random.normal(loc=10,scale=5,size=size_rand[0])
c = np.random.normal(loc=10,scale=5,size=1)

sol = x + c[0]*y

spio.mmwrite("x_axy",x[:,None])
spio.mmwrite("y_axy",y[:,None])
spio.mmwrite("c_axy",c[:,None])
spio.mmwrite("sol_axy",sol[:,None])

