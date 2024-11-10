import numpy as np
import time

def somma(n):
    for i in range(n):
        n += i
    return n

def sommang(n):
    return (n + 1) * (n / 2) # usa la formula, più rapido e tempi consistenti
    
i = int(input())

t = time.time()
a = somma(i)
print(a, time.time() - t)

t = time.time()
a = sommang(i)
print(a, time.time() - t)