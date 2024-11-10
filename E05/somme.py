import numpy as np
import math


def sommang(n):
    return (n + 1) * (n / 2) # usa la formula, più rapido e tempi consistenti

def sommaRad(n):    # costa più memoria, ma ha tempi consistenti e inferiori sui grandi numeri
    m = np.sqrt(np.arange(1, n + 1))
    return np.sum(m)

def sommaRadApprox(n):    # l'approssimazione alla forma scende di imprecisione sotto lo 0.5% sopra il 150
    if n < 150:
        m = np.sum(np.sqrt(np.arange(1,n+1)))
    else:
        m = (2/3) * (n**1.5)
    return m

def sumNDot(n):
    p = math.factorial(n)
    s = (n + 1) * (n / 2)
    return s, p

def sumPow(n, *k):
    if len(k) > 0 and k[0] != 1:
        m = 0
        for i in range(1, n + 1):
            m += i**k[0]
        return m
    else:
        return (n + 1) * (n / 2)