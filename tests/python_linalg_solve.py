import random

import numpy as np
from scipy import io as spio


size_rand_0 = random.randint(50,100)
size_rand_1 = size_rand_0 - random.randint(30,50)
A = np.random.normal(loc=10,scale=5,
                     size=(size_rand_0,size_rand_1))
b = np.random.normal(loc=10,scale=5,size=size_rand_0)


ATA = A.T@A
ATb = A.T@b
sol = np.linalg.solve(ATA, ATb)

spio.mmwrite("A_solve",A)
spio.mmwrite("b_solve",b[:,None])
spio.mmwrite("sol_solve",sol[:,None])
             

