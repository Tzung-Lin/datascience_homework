import numpy as np

a=np.array([1,2,3,4,5])
b=np.array([2,2,2,2,2])

print(np.reshape(np.hstack((a,b)),(5,2)))