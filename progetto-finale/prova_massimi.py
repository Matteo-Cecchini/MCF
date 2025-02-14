import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def shuffle(signal, shuffles = 100):

    dummy = signal.copy()
    maxs = np.zeros(shuffles) # maxs o maxes?
    
    # utilizzare map() non comporta ottimizzazioni
    for i in range(shuffles):
        np.random.shuffle(dummy)
        maxs[i] = max(np.abs(fft(dummy)[1:])**2) # il primo coefficiente è l'offset
    
    # il percentile ha significato a prescindere dalla distribuzione, i masismi non sono distribuiti normalmente
    return maxs