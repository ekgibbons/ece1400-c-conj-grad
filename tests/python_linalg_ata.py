import numpy as np
from scipy import io as spio

size_rand = np.random.randint(7,11,size=(2))
A = np.random.normal(loc=10,scale=5,size=size_rand)

sol = A.T@A

spio.mmwrite("A_ata",A)
spio.mmwrite("sol_ata",sol,symmetry="general")

