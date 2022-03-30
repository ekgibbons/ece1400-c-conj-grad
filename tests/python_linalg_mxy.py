import numpy as np
from scipy import io as spio

size_rand = np.random.randint(7,11,size=(1))
x = np.random.normal(loc=10,scale=5,size=size_rand[0])
y = np.random.normal(loc=10,scale=5,size=size_rand[0])

spio.mmwrite("x_dax",x[:,None])
spio.mmwrite("y_dax",y[:,None])

sol = y.T@x

print("%.16e" % sol)

