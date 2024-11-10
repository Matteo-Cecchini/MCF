import numpy as np
import time

def sommang(n):
    return (n + 1) * (n / 2) # usa la formula, più rapido e tempi consistenti

def sommaRad(n):    # l'approssimazione alla forma integrale si discosta dello 0.5% sopra 150
    if n < 150:
        m = np.sum(np.sqrt(np.arange(1,n+1)))
    else:
        m = (2/3) * (n**1.5)
    return m
    
i = int(input())

print(sommaRad(i))