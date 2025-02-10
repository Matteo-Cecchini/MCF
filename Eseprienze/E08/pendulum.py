import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

G = 9.80665

def pfun(r, t, l=.5):
    dtheta = r[1]
    domega = -(G/l)*np.sin(r[0])
    ddt = [dtheta, domega]
    return ddt

rad = lambda x: np.pi*x/180
l = [.5, 1, .5]
r0 = ([rad(45), 0], [rad(45), 0], [rad(30), 0])

time = np.arange(0, 10, 0.01)

sols = [scipy.integrate.odeint(pfun, i, time, args=(j,)) for i,j in zip(r0, l)]
print(sols)
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].plot(time, sols[0], label=["Angolo", "Vel. Angolare"])
ax[1].plot(time, sols[1], label=["Angolo", "Vel. Angolare"])
ax[2].plot(time, sols[2], label=["Angolo", "Vel. Angolare"])
ax[0].legend()
ax[1].legend()
ax[2].legend()
'''
Gli angoli più piccoli comportano una magnitudine massima di velocità angolare inferiore, dacché 
il pendolo non viene accelerato "verso il basso" come succede per angoli più grandi. 

Lunghezze maggiori del pendolo amplificano il periodo di oscillazione, 
come si vede nei primi due plot a parità di angolo.

'''

plt.show()