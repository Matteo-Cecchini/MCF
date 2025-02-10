import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import funs

# colonne utili all'esercitazione provenienti dal seguente csv: http://opendata.cern.ch/record/5203/files/Jpsimumu.csv
dataset = pd.read_csv("Jpsimumu.csv")
cols = dataset.columns

pairs = list(map(lambda v, w: (dataset[v] + dataset[w])**2, cols[::2], cols[1::2]))
inv_energy = np.sqrt(pairs[0] - np.sum(pairs[1:], axis=0))


fig, (ax1, ax2) = plt.subplots(1,2)
heights, edges, bars = ax1.hist(inv_energy, bins=100, color="darkturquoise")

zoom = 0.25
edge_peak = np.where(heights == np.max(heights))[0][0]
inv_energy_zm = [i for i in inv_energy if (i - edges[edge_peak])**2 < zoom**2]
heights_zm, edges_zm, bars = ax2.hist(inv_energy_zm, bins=100, color="tomato")

plt.show()


bins = (edges[:-1] + edges[1:]) / 2
bins_zm = (edges_zm[:-1] + edges_zm[1:]) / 2

par1G, par1G_cov = optimize.curve_fit(funs.gaussNpoly, bins, heights, p0=[1]*5)
par2G, par2G_cov = optimize.curve_fit(funs.doubleGaussNpoly, bins, heights, p0=[2]*7)

h1G = funs.gaussNpoly(bins, *par1G)
h2G = funs.doubleGaussNpoly(bins, *par2G)

v1G = np.sum( np.sqrt(np.diag(par1G_cov)) )
v2G = np.sum( np.sqrt(np.diag(par2G_cov)) )

d1G = heights - h1G
d2G = heights - h2G

c1G = d1G / v1G
c2G = d1G / v2G

fig, ax = plt.subplots(2,3)
ax = ax.flatten()

ax[0].hist(inv_energy, bins=100)
ax[0].plot(bins, h1G)

ax[1].scatter(bins, d1G)
ax[2].scatter(bins, c1G)

ax[3].hist(inv_energy, bins=100)
ax[3].plot(bins, h2G)

ax[4].scatter(bins, d2G)
ax[5].scatter(bins, c1G)

plt.show()

c1G, c2G = np.sum(c1G**2)/len(c1G), np.sum(c2G**2)/len(c2G)
print(c1G, c2G)