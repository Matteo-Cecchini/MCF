import LCRanalysis as LCR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import norm
from time import time
import os

directory = "J0137"
paths = [os.path.join(directory, i) for i in os.listdir(directory)]

a, b = [LCR.read_csv("lcr", i) for i in paths]


def plot_data(d1: LCR.Datasheet, d2: LCR.Datasheet, n, sigmas, timeformat: str = "", see_shuffle: bool = False):
    # Pre-processing for d1
    if len(d1.significativity) == 0:
        d1.shuffle_analysis(inplace=True, n=n, sigmas=sigmas)
    if len(d1.frequencies) == 0:
        d1.FFT(inplace=True)
    
    # Pre-processing for d2
    if len(d2.significativity) == 0:
        d2.shuffle_analysis(inplace=True, n=n, sigmas=sigmas)
    if len(d2.frequencies) == 0:
        d2.FFT(inplace=True)
    
    # Set cut based on time data length for d1 and d2
    cut1 = len(d1.timedata) // 2
    cut2 = len(d2.timedata) // 2
    sig_freq_d1, sig_pows_d1 = d1.frequencies, np.absolute(d1.coefficients)**2
    sig_freq_d2, sig_pows_d2 = d2.frequencies, np.absolute(d2.coefficients)**2
    
    # Significant frequency power calculation for d1
    indexes_d1 = [i for i in d1.significativity["indexes"] if i < cut1]
    sig_freq_d1 = d1.frequencies[indexes_d1]
    sig_pows_d1 = np.absolute(d1.coefficients[indexes_d1])**2
    
    # Significant frequency power calculation for d2
    indexes_d2 = [i for i in d2.significativity["indexes"] if i < cut2]
    sig_freq_d2 = d2.frequencies[indexes_d2]
    sig_pows_d2 = np.absolute(d2.coefficients[indexes_d2])**2
    
    # Calculate percentage for significance
    perc = norm.cdf(sigmas)*100
    
    # Create the figure with two subplots (1 row, 2 columns)
    fig = plt.figure(figsize=(13,6))
    gs = fig.add_gridspec(1, 2)
    
    # Left subplot: Power spectrum for d1 and d2
    ps, sn = gs.subplots()
    
    # Plot for d1
    ps.plot(sig_freq_d1, sig_pows_d1, ".", lw=1, label=f"Picchi sett. significativi d1 oltre {round(perc, 3)}%")
    ps.plot(d1.frequencies[:cut1], np.absolute(d1.coefficients[:cut1])**2, lw=1, color="gray", alpha=.5, label="Spettro Potenza FFT settimanali")
    ps.hlines(np.absolute(d1.significativity["mean"] + sigmas*d1.significativity["std"]), 0, np.max(d1.frequencies[:cut1]), color="darkturquoise", alpha=.4, label=f"Media + {round(sigmas, 3)} deviazioni (sett.)", linestyles="dashed")
    
    # Plot for d2
    ps.plot(sig_freq_d2, sig_pows_d2, ".", lw=1, label=f"Picchi mens. significativi d2 oltre {round(perc, 3)}%")
    ps.plot(d2.frequencies[:cut2], np.absolute(d2.coefficients[:cut2])**2, lw=1, color="wheat", alpha=1, label="Spettro Potenza FFT mensili")
    ps.hlines(np.absolute(d2.significativity["mean"] + sigmas*d2.significativity["std"]), 0, np.max(d2.frequencies[:cut2]), color="plum", alpha=.4, label=f"Media + {round(sigmas, 3)} deviazioni (mens.)", linestyles="dashed")
    
    # Add vertical lines for significant frequencies
    for i in range(len(indexes_d1)):
        ps.annotate("w {:.2e}".format(sig_freq_d1[i]), xy=(sig_freq_d1[i], sig_pows_d1[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    for i in range(len(indexes_d2)):
        ps.annotate("m {:.2e}".format(sig_freq_d2[i]), xy=(sig_freq_d2[i], sig_pows_d2[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    
    #lims = [min(np.min(sig_freq_d1), np.min(sig_freq_d2)), max(np.max(sig_freq_d1), np.max(sig_freq_d2))]
    lims = max(d2.frequencies[cut2 - 1], max(sig_freq_d1))
    ps.set_xlim(0, lims)
    ps.set_title("Spettro di potenza delle due strutture")
    ps.set_xlabel("Frequenza [Hz]")
    ps.set_ylabel("Potenza (log)")
    ps.set_yscale("log")
    ps.legend()
    
    # Right subplot: Data synthesis and plot for d2
    cc_d1 = np.zeros(len(d1.coefficients), dtype=np.complex128)
    cc_d1[indexes_d1] = d1.coefficients[indexes_d1]
    yy_d1 = np.absolute(ifft(cc_d1))
    
    # Right subplot: Data synthesis and plot for d2
    cc_d2 = np.zeros(len(d2.coefficients), dtype=np.complex128)
    cc_d2[indexes_d2] = d2.coefficients[indexes_d2]
    yy_d2 = np.absolute(ifft(cc_d2))

    # Format time data for the plot
    if d1.csvformat["name"] == "lcr" and d2.csvformat["name"] == "lcr":
        if timeformat.lower() in ("met", "mission elapsed time"):
            timelabel = "Mission Elapsed Time (MET)"
            timedata_d1 = 86400*(d1.timedata - d1.timedata[0]) + d1.csvformat["MET0"]
            timedata_d2 = 86400*(d2.timedata - d2.timedata[0]) + d2.csvformat["MET0"]
        else:
            print("eccomi")
            timelabel = d1.nameofdata[0]
            timedata_d1 = d1.timedata
            timedata_d2 = d2.timedata
          
    if d2.limitmask is not None:
        notmask = ~d2.limitmask
        sn.scatter(timedata_d2[d2.limitmask], d2.fluxdata[d2.limitmask], marker="v", label="Upper Limit (mens.)")
    else:
        notmask = np.full(d2.timedata.shape, True)
    
    sn.errorbar(timedata_d2[notmask], d2.fluxdata[notmask], d2.sigmadata[notmask], 
                    fmt=".", elinewidth=1, ecolor="gray", label="Detection (mens.)")
    sn.plot(timedata_d2[notmask], d2.fluxdata[notmask], lw=1, color="gray", alpha=.5)

    # Plot for d2
    sn.plot(timedata_d1, yy_d1, lw=1, color="darkturquoise", label=f"Sintesi picchi significativi oltre {perc.round(2)} (sett.)%")
    sn.plot(timedata_d2, yy_d2, lw=1, color="plum", label=f"Sintesi picchi significativi oltre {perc.round(2)}% (mens.)")
    if see_shuffle:
        sn.plot(timedata_d2, d2.significativity["data"], lw=1, color="#1f77b4", alpha=.5, label="Media curve sintetiche d2")
    sn.set_xlabel(timelabel)
    sn.set_ylabel(d2.nameofdata[1])
    sn.set_title(f"Plot sintesi dati con picchi significativi oltre {round(perc, 3)}%,\na confronto con la curva di luce dei dati mensili")
    sn.legend()
    
    # Show the final plot
    plt.tight_layout()
    plt.show()
    
plot_data(a, b, 100, 3.9)
