import scipy.stats
import reco
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy


csv_list = ("hit_times_M0.csv", "hit_times_M1.csv", "hit_times_M2.csv", "hit_times_M3.csv")
hit_arr = [reco.array_hit(i) for i in csv_list]

hit_arr = np.concatenate(hit_arr)
hit_arr = np.sort(hit_arr)


dtime = np.array(hit_arr[1:] - hit_arr[:-1])

# istogramma parte uno
'''
fig, ax = plt.subplots(1,1, figsize=(12,9))

ax.hist(x=dtime, bins=100, color="gold")
ax.set_xlabel("differenze temporali tra hit consecutivi (ns)")
ax.set_ylabel("frequenza per bin")
ax.set_title("istogramma $\Delta$t in nanosecondi")

plt.show()
'''

events = reco.array_event(hit_arr)
    
'''
for i in events[:10]:
    print()
    i.print()

nums = [i.hit_num for i in events]
times = np.array([i.time_lapse for i in events])
print(times)
delta_times = times[1:] - times[:-1]

fig, ax = plt.subplots(2,2, figsize=(10,7))

ax[0][0].hist(nums, bins=100, color="limegreen")
ax[0][0].set_xlabel("Numero di hit per evento")
ax[0][0].set_ylabel("frequenze")
ax[0][0].set_title("Istogramma numeri di hit per evento")

ax[0][1].hist(times, bins=100, color="gold")
ax[0][1].set_xlabel("Durata di tempo per evento")
ax[0][1].set_ylabel("frequenze")
ax[0][1].set_title("Istogramma surata di tempo per evento")

ax[1][0].hist(delta_times, bins=100, color="darkturquoise")
ax[1][0].set_xlabel("Numero di hit per evento")
ax[1][0].set_ylabel("frequenze")
ax[1][0].set_title("Istogramma numeri di hit per evento")

ax[1][1].scatter(times, nums, color="tomato")
ax[1][1].set_xlabel("Tempi di durata dell'evento (ns)")
ax[1][1].set_ylabel("Numero di hit per evento")
ax[1][1].set_title("Scatter di numero di hit per evento in funzione della durata")

plt.tight_layout()
plt.show()
'''

fig, axes = plt.subplots(2,5, figsize=(21, 7))
axes = axes.flatten()

# Coordinate centro Moduli [m]
xmod = [-5,  5, -5,  5]
ymod = [ 5,  5, -5, -5]

# Coordinate dei Sensori rispetto al centro del Modulo [m]
xdet = [-2.5, 2.5, 0, -2.5,  2.5]
ydet = [ 2.5, 2.5, 0, -2.5, -2.5]

coordx = []
coordy = []
for i in range(4):
    coordx.append([xmod[i] + xdet[j] for j in range(5)])
    coordy.append([ymod[i] + ydet[j] for j in range(5)])

for i, ax in enumerate(axes):
        hit_x = [xmod[k.mod_id] + xdet[k.det_id] for k in events[i]]
        hit_y = [ymod[k.mod_id] + ydet[k.det_id] for k in events[i]]
        hit_times = [k.hit_time - events[i].first_hit for k in events[i]]
        
        scatter = ax.scatter(hit_x, hit_y, s=240, c=hit_times, cmap="plasma")
        scatter.set_clim(0,150)
        fig.colorbar(scatter, ax=ax, label="Hit t-t_{start}")
        ax.scatter(coordx, coordy, s=240, facecolors="none", edgecolors="black", alpha=0.1)
        ax.axhline(0, color="black", alpha=0.25)
        ax.axvline(0, color="black", alpha=0.25)
        ax.set_xlabel("Coordinate x dei Sensori [m]")
        ax.set_ylabel("Coordinate y dei Sensori [m]")
        ax.set_xticks(np.arange(-10, 12.5, 2.5))
        ax.set_yticks(np.arange(-10, 12.5, 2.5))
        ax.set_title(f"Geometria hit evento n°{i} ")

plt.tight_layout()
plt.show()