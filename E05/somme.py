import numpy as np
import time

def somma(n):
    for i in range(n):
        n += i
    return a

def sommang(n):
    a = np.array(1,n)
    return np.sum(a)
    
i = int(input())

t = time.time()
a = somma(i)
print(a, time.time() - t)

t = time.time()
a = sommang(i)
print(a, time.time() - t)