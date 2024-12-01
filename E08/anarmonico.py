import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as si

def osc(r, t, w = 1):
    dx = r[1]
    dy = -(w**2)*r[0]**3
    ddt = [dx, dy]
    return ddt

r0 = ([0,0], [1,0], [0,1], [1,1])
time = np.linspace(0,10,1000)

sols = [si.odeint(osc, i, time) for i in r0]

fig, axes = plt.subplots(2,2, figsize=(10,10))
for ax,sol in zip(axes.flat, sols):
    ax.plot(time, sol, label=("Posizione", "Velocità"))
    ax.legend()
plt.show()