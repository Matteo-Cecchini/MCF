from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# plot
csvs = ("sample1.csv", "sample2.csv", "sample3.csv")
data = [pd.read_csv(i) for i in csvs]
data = [i[i.columns[1]].values for i in data]

fig, axs = plt.subplots(3,1, sharex=True)
for i,j,k in zip(axs.flat, data, csvs):
    i.plot(j)
    i.set_xlabel("time")
    i.set_ylabel("meas")
    i.set_title(k)
plt.tight_layout()
plt.show()

# power spectrum
fftcs = [np.absolute(rfft(i))**2 for i in data]
freqs = [rfftfreq(len(i)) for i in fftcs]

fig, axs = plt.subplots(1,3)
for i,j,k in zip(axs, freqs, fftcs):
    i.plot(j,k[:len(j)], 'o')
plt.show()

def noise(x, a, b):
    return a/(x**b)
pws = ((1,0), (.1,1), (.5,1))

pars = [curve_fit(noise, i[1:], np.absolute(j[1:len(i)])**2, p0=k, maxfev=1000) for i,j,k in zip(freqs, fftcs, pws)]
ys = [noise(i, j[0][0], j[0][1]) for i,j in zip(freqs, pars)]


fig, axs = plt.subplots(1,3)
for i,j,k,l in zip(axs, freqs, fftcs, ys):
    i.plot(j,k[:len(j)], 'o')
    i.plot(j,l)
plt.show()

"""
I tre rumori sembrano essere rispettivamente bianco, rosa e rosso. La difficoltà del fit sta nell' immettere parametri iniziali
che possono ottimizzare la funzione; l'ottimizzazione infatti sembra essere vittima di una sensibilità estrema ai parametri iniziali.
"""