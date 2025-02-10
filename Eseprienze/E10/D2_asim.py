from randomwalk import cum, random_walk
import matplotlib.pyplot as plt
import numpy as np

def p(x):
    return np.sin(x/2)/4

def inv(y):
    return 2*np.arccos(1 - 2*y)

pos = [cum(inv, 1000, const=(1,0)) for i in range(5)]

for i, j in zip(pos, range(len(pos))):
    plt.plot(i[0], i[1], label=f"Random Walker {j+1}")
plt.legend()
plt.show()

fig, axs = plt.subplots(1,2, figsize=(10,5))
for i, j in zip(pos, range(len(pos))):
    axs[0].plot(i[0], i[1], label=f"Random Walker {j+1}")
    axs[1].plot(i[0]**2 + i[1]**2, label=f"|(x,y)|^2 del Ran.Walker {j+1}")
plt.legend()
plt.show()