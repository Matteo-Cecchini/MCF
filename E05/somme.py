import numpy as np
import time

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