import multiprocessing.process
import LCRanalysis as LCR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import norm
from time import time
import os
import multiprocessing

def shuffle_analysis(signal, iterations, percentile):
    dummy = signal.copy()
    maxs = np.zeros(iterations) # maxs o maxes?
    
    for i in range(iterations):
        dummy = np.random.permutation(signal)
        maxs[i] = max(np.abs(fft(dummy)[1:])**2) # il primo coefficiente è l'offset
    
    treshold = np.percentile(maxs, percentile)
    indexes = np.where(np.abs(fft(signal))**2 > treshold)[0]
    return {"treshold":treshold, "percentile":percentile, "indexes":indexes}   
    

a = LCR.read_csv("lcr", "J0137/4FGL_J0137.0+4751_weekly_12_27_2024.csv")
a.convert_to_numeric(inplace=True)

b = a.fluxdata
c = shuffle_analysis(b, 100, 95)
print(c)

k = fft(b)
f = fftfreq(len(k), d=7)

plt.plot(f[:len(b)//2], np.absolute(k[:len(b)//2])**2, c="gray", lw=1)
plt.hlines(c[0], 0, .05)
plt.yscale('log')
plt.show()