import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.fft import ifft
import LCRanalysis as LCR
import os

def load_data(filepath):
    data = LCR.read_csv("lcr", filepath)
    return data

def analyze_periodicity(data: LCR.Datasheet, sigmas = 3.9, num_shuffles=100):
    """Effettua l'analisi di Fourier e verifica la periodicità."""
    data.FFT(inplace=True)
    data.shuffle_analysis(n=num_shuffles, sigmas=sigmas, inplace=True)

def plot_results(data: LCR.Datasheet, choice, sigmas, timeformat):
    """Genera i grafici della curva di luce e spettro di potenza."""
    if choice == "analysis":
        data.plot_analysis(sigmas=sigmas)
    elif choice == "all":
        data.plot_data(timeformat)
        data.plot_spectrum(see_parts=True)
        data.plot_analysis(sigmas=sigmas, see_shuffle=True, timeformat=timeformat)
    elif choice == "data":
        data.plot_data(timeformat)
    elif choice == "spectrum":
        data.plot_spectrum(see_parts=True)

def confront_and_plot(d1: LCR.Datasheet, d2: LCR.Datasheet, n, sigmas, names, timeformat: str = ""):
    # Controllo che ci sia tutto in d1
    d1.convert_to_numeric(True, True)
    d1.FFT(True)
    d1.shuffle_analysis(n, sigmas, True)
    
    d2.convert_to_numeric(True, True)
    d2.FFT(True)
    d2.shuffle_analysis(n, sigmas, True)
    
    # DIstinzione delle lunghezze degli array di frequenza e coefficienti FFT
    cut1 = len(d1.timedata) // 2
    cut2 = len(d2.timedata) // 2
    sig_freq_d1, sig_pows_d1 = d1.frequencies, np.absolute(d1.coefficients)**2
    sig_freq_d2, sig_pows_d2 = d2.frequencies, np.absolute(d2.coefficients)**2
    
    # Prendo i dati sopra la soglia di significatività scelta
    indexes_d1 = [i for i in d1.significativity["indexes"] if i < cut1]
    sig_freq_d1 = d1.frequencies[indexes_d1]
    sig_pows_d1 = np.absolute(d1.coefficients[indexes_d1])**2
    
    indexes_d2 = [i for i in d2.significativity["indexes"] if i < cut2]
    sig_freq_d2 = d2.frequencies[indexes_d2]
    sig_pows_d2 = np.absolute(d2.coefficients[indexes_d2])**2
    
    # Calcolo il valore percentuale relativo al numero di deviazioni scelto
    perc = norm.cdf(sigmas)*100
    fontsize = 7.5
    
    # Definizione figura
    fig = plt.figure(figsize=(13,6))
    gs = fig.add_gridspec(1, 2)
    
    # Plot a sinistra: spettro di potenza a confronto dati settimanali e mensili
    ps, sn = gs.subplots()
    
    # settimanali
    ps.plot(sig_freq_d1, sig_pows_d1, ".", lw=1, label=f"Picchi sett. significativi d1 oltre {round(perc, 3)}%")
    ps.plot(d1.frequencies[:cut1], np.absolute(d1.coefficients[:cut1])**2, lw=1, color="gray", alpha=.5, label=f"Spettro Potenza {names[0]}")
    ps.hlines(np.absolute(d1.significativity["mean"] + sigmas*d1.significativity["std"]), 0, np.max(d1.frequencies[:cut1]), color="darkturquoise", alpha=.4, label=f"Media + {round(sigmas, 3)} deviazioni (sett.)", linestyles="dashed")
    
    # mensili
    ps.plot(sig_freq_d2, sig_pows_d2, ".", lw=1, label=f"Picchi mens. significativi d2 oltre {round(perc, 3)}%")
    ps.plot(d2.frequencies[:cut2], np.absolute(d2.coefficients[:cut2])**2, lw=1, color="wheat", alpha=1, label=f"Spettro Potenza {names[1]}")
    ps.hlines(np.absolute(d2.significativity["mean"] + sigmas*d2.significativity["std"]), 0, np.max(d2.frequencies[:cut2]), color="plum", alpha=.4, label=f"Media + {round(sigmas, 3)} deviazioni (mens.)", linestyles="dashed")
    
    # Linee verticali dei coefficienti oltre la soglia di significatività
    '''
    for i in range(len(indexes_d1)):
        ps.annotate("w {:.2e}".format(sig_freq_d1[i]), xy=(sig_freq_d1[i], sig_pows_d1[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    for i in range(len(indexes_d2)):
        ps.annotate("m {:.2e}".format(sig_freq_d2[i]), xy=(sig_freq_d2[i], sig_pows_d2[i]), xytext=(5, 5), textcoords="offset points", ha="left", va="bottom", fontsize=8, color="black", fontweight='light')
    '''
    # zoom o su tutto l'array delle potenze FFT dei mensili o fino l'ultima potenza significativa dei settimanali
    lims = max(d2.frequencies[cut2 - 1], max(sig_freq_d1))
    ps.set_xlim(0, lims)
    ps.set_title("Spettro di potenza delle due strutture")
    ps.set_xlabel("Frequenza [Hz]")
    ps.set_ylabel("Potenza (log)")
    ps.set_yscale("log")
    ps.legend(fontsize=fontsize)
    
    # Plot a detra: sintesi a confronto con i dati mensili (sono di meno e si vedono di più)
    # sintesi settimanali
    cc_d1 = np.zeros(len(d1.coefficients), dtype=np.complex128)
    cc_d1[indexes_d1] = d1.coefficients[indexes_d1]
    yy_d1 = np.absolute(ifft(cc_d1))
    
    # sintesi mensili
    cc_d2 = np.zeros(len(d2.coefficients), dtype=np.complex128)
    cc_d2[indexes_d2] = d2.coefficients[indexes_d2]
    yy_d2 = np.absolute(ifft(cc_d2))

    # Formattazione valori temporali se si vuole Julian Date o MET
    if d1.csvformat["name"] == "lcr" and d2.csvformat["name"] == "lcr":
        if timeformat.lower() in ("met", "mission elapsed time"):
            timelabel = "Mission Elapsed Time (MET)"
            timedata_d1 = 86400*(d1.timedata - d1.timedata[0]) + d1.csvformat["MET0"]
            timedata_d2 = 86400*(d2.timedata - d2.timedata[0]) + d2.csvformat["MET0"]
        else:
            timelabel = d1.nameofdata[0]
            timedata_d1 = d1.timedata
            timedata_d2 = d2.timedata

    # vede se in d2 c'è limitmask, altrimenti non fa distinzione con upper limit. potrei toglierla sapendo che non c'è possibilita di scelta in questo script ma non si sa mai
    if d2.limitmask is not None:
        notmask = ~d2.limitmask
        sn.scatter(timedata_d2[d2.limitmask], d2.fluxdata[d2.limitmask], marker="v", label="Upper Limit (mens.)")
    else:
        notmask = np.full(d2.timedata.shape, True)
    
    sn.errorbar(timedata_d2[notmask], d2.fluxdata[notmask], d2.sigmadata[notmask], 
                    fmt=".", elinewidth=1, ecolor="gray", label="Detection (mens.)")
    sn.plot(timedata_d2[notmask], d2.fluxdata[notmask], lw=1, color="gray", alpha=.5)

    # plot
    sn.plot(timedata_d1, yy_d1, lw=1, color="darkturquoise", label=f"Sintesi picchi significativi oltre {perc.round(3)} (sett.)%")
    sn.plot(timedata_d2, yy_d2, lw=1, color="plum", label=f"Sintesi picchi significativi oltre {perc.round(3)}% (mens.)")

    sn.set_xlabel(timelabel)
    sn.set_ylabel(d2.nameofdata[1])
    sn.set_title(f"Plot sintesi dati con picchi significativi oltre {round(perc, 3)}%,\na confronto con la curva di luce dei dati mensili")
    sn.legend(fontsize=fontsize)
    
    # show
    plt.tight_layout()
    plt.show()

def do_things(parser):
    args = parser.parse_args()
    if args.dir:
        names = os.listdir(args.filepath)
        paths = [os.path.join(args.filepath,i) for i in names] # qui assumo che ci siano solo due file nella cartella, estendibile a n file anche nella funzione
        paths = paths if "weekly" in names[0] else paths[::-1] # inversione se legge monthly prima di weekly
        names = names if "weekly" in names[0] else names[::-1] # inversione se legge monthly prima di weekly
        data1, data2 = [load_data(i) for i in paths]
        confront_and_plot(data1, data2, args.shuffles, args.sigmas, names, args.timeformat)
    else:
        data = load_data(args.filepath)
        choice = args.plot
        
        if args.percentage is not None:
            args.sigmas = norm.ppf(args.percentage / 100)
        
        analyze_periodicity(data, sigmas=args.sigmas, num_shuffles=args.shuffles)
        
        if args.timeformat is not None and args.timeformat.lower() in ("met", "mission elapsed time"):
            args.timeformat == "met"
            
        plot_results(data, choice, args.sigmas, args.timeformat)
        

def main():
    parser = argparse.ArgumentParser(description="Analisi di periodicità delle curve di luce dei Blazar")
    parser.add_argument("filepath", type=str, help="Percorso del file CSV della curva di luce")
    parser.add_argument("-d", "--dir", type=bool, default=False, help="Se è una cartella (True) mette a confronto tutti i file nella cartella")
    parser.add_argument("-p", "--plot", choices=["all", "data", "spectrum", "analysis"], default="analysis", help="Scelta del tipo di plot")
    parser.add_argument("-n", "--shuffles", type=int, default=1000, help="Numero di curve di luce sintetiche")
    parser.add_argument("-s", "--sigmas", type=float, default=3.9, help="Soglia per la significatività del picco in deviazioni")
    parser.add_argument("-c", "--percentage", type=float, help="Soglia per la significatività del picco in percentuale")
    parser.add_argument("-t", "--timeformat", choices=["JD", "Julian Date", "MET", "Mission Elapsed Time"], 
                        default="JD", help="Scelta del formato data, Julian Date o MET")
    
    do_things(parser)

if __name__ == "__main__":
    main()