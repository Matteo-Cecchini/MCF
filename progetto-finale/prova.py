import multiprocessing.process
import LCRanalysis as LCR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from time import time
import os
import multiprocessing as mp
from functools import partial

# tentativi di perallelizzare lo shuffle

def shuffle1(signal, iterations, percentile):
    dummy = signal.copy()
    maxs = np.zeros(iterations) # maxs o maxes?
    
    for i in range(iterations):
        dummy = np.random.permutation(signal)
        maxs[i] = max(np.abs(fft(dummy)[1:])**2) # il primo coefficiente è l'offset

    treshold = np.percentile(maxs, percentile)
    indexes = np.where(np.abs(fft(signal))**2 > treshold)[0]
    return {"treshold":treshold, "percentile":percentile, "indexes":indexes}
'''
def find_max(signal):
        dummy = np.random.permutation(signal)  
        return max(np.abs(fft(dummy)[1:])**2)

def shuffle2(signal, iterations = 100, percentile = 95):
    
    with mp.Pool(processes=4) as pool:
        maxs = pool.imap(find_max, (signal for _ in range(iterations)), chunksize=10)  # Non duplico memoria
    maxs = list(maxs)
    
    treshold = np.percentile(maxs, percentile)
    indexes = np.where(np.abs(fft(signal))**2 > treshold)[0]
    return {"treshold": treshold, "percentile": percentile, "indexes": indexes}
'''

a = LCR.read_csv("lcr", "J1256/4FGL_J1256.1-0547_weekly_12_27_2024.csv")