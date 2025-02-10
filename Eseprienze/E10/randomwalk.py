import numpy as np
from itertools import accumulate
from scipy.integrate import simpson

TAU = 2*np.pi

def random_walk(num: int = 100, passo: float = 1):
    if num < 0:
        num = -num
    
    phi = np.random.uniform(low=0, high=TAU, size=num)
    dX = passo*np.cos(phi)
    dY = passo*np.sin(phi)
    dX[0] = 0
    dY[0] = 0
    x = list(accumulate(dX))
    y = list(accumulate(dY))
    return np.array([x, y])

def cum(func: callable, num: int = 100, passo: float = 1, const: tuple[float,float] = (0,0)):
    ps = np.random.random(num)
    phis = func(ps)
    dX = passo*np.cos(phis) + const[0]
    dY = passo*np.sin(phis) + const[1]
    dX[0] = 0
    dY[0] = 0
    x = list(accumulate(dX))
    y = list(accumulate(dY))
    return np.array([x, y])