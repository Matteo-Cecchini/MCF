import numpy as np
import randomwalk as rw
import matplotlib.pyplot as plt

pos5 = [rw.random_walk(1000) for i in range(5)]

for i, j in zip(pos5, range(len(pos5))):
    plt.plot(i[0], i[1], label=f"Random Walker {j+1}")
plt.legend()
plt.show()

pos = [rw.random_walk(1000) for i in range(100)]

fig, axs = plt.subplots(1,3, figsize=(15,5))
for i, j in zip(pos, range(len(pos))):
    axs[0].plot(i[0][:10], i[1][:10])
    axs[1].plot(i[0][:100], i[1][:100])
    axs[2].plot(i[0], i[1])
plt.show()

fig, axs = plt.subplots(1,2, figsize=(10,5))
for i, j in zip(pos5, range(len(pos))):
    axs[0].plot(i[0], i[1], label=f"Random Walker {j+1}")
    axs[1].plot(i[0]**2 + i[1]**2, label=f"|(x,y)|^2 del Ran.Walker {j+1}")
plt.legend()
plt.show()