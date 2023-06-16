import numpy as np
from utilities import transposon_operator

P, d = 10, 3

ep = np.random.randint(1, 22, (P, d))

print(ep)
print('----------------------------')
print(transposon_operator(ep, .7))