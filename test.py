
import numpy as np


depth =3
width = 4
mask = np.zeros((depth, width //2, 2))
for i in range(1, depth, 2):  # odd layers only
    mask[i] = 1.0

print(mask)