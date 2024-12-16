import serie
import numpy as np
import time

def fibo(n):
    fib = 1
    fob = 1
    if n > 2:
        for i in range(2,n):
            fib, fob = fib + fob, fib
    return fib/fob

n = 10000

t = time.time()
a = serie.fibonacci(n)
t1 = time.time() - t
t = time.time()
b = fibo(n)
t2 = time.time() - t
print(a, t1)
print(b, t2)

